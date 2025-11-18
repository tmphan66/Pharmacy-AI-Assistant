import os
import re
import pandas as pd
import easyocr
import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor, tool
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentFinish
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. Configurations ---
CSV_PATH = "data/drugs_cleaned.csv"
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:4b"

print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'])
print("EasyOCR reader initialized.")

# --- 2. Build ChromaDB function ---
def build_chroma_db():
    """
    Build a new ChromaDB from the CSV ONCE. Persists to DB_PATH.
    """
    print(f"--- Building new ChromaDB index at {DB_PATH} ---")
    print(f"Loading dataset from '{CSV_PATH}'...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['review', 'drugName', 'condition', 'rating'])
    print(f"Loaded with {len(df)} records.")

    print("Converting data to LangChain Documents...")
    documents = []
    for _, row in df.iterrows():
        drug = str(row['drugName']).strip()
        cond = str(row['condition']).strip()
        review = str(row["review"])
        sentiment = str(row['sentiment_label']).strip()

        page_content = f"Drug: {drug}. Condition: {cond}. Review: {review}"
        
        metadata = {
            "drugName": drug,
            "condition": cond,
            "rating": int(row['rating']),
            "sentiment_label": sentiment
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # Use "cuda" if you have a GPU
    )
    print("Embedding model loaded.")

    print("Building ChromaDB index... This may take a moment...")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("ChromaDB index built successfully.")
    print(f"Chroma vector database saved to {DB_PATH} folder.")
    print("--- Building ChromaDB completed. ---")

# --- 3. Agent Loading function ---
def load_agent_system():
    """
    Check for DB, load models, and create the agent executor.
    """
    if not os.path.isdir(DB_PATH):
        print(f"ChromaDB not found at '{DB_PATH}'. Building new database...")
        build_chroma_db()

    # Load LLM
    print(f"Loading LLM model ({LLM_MODEL})...")
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1, max_tokens=1024)

    # Load embedding model
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # Use "cuda" if you have a GPU
    )

    # Load ChromaDB
    print("Loading Chroma DB...")
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    # Load CSV into Pandas (for tools that compute over the table)
    print("Loading CSV into Pandas...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['review', 'drugName', 'condition', 'rating'])
    df['rating'] = df['rating'].astype(int)
    print("--- All models and data loaded successfully. ---")

    # --- 4. Define Tools ---
    simple_retriever = vector_db.as_retriever(
        search_type="similarity", 
        search_kwargs={"k":5} # Change k for more/fewer results depending on your needs
    )

    @tool
    def process_image(query_about_image: str) -> str:
        """
        Extracts text from an uploaded image using OCR.

        - Call this tool AT MOST ONCE per user question.
        - Only use it if the user uploaded an image and you need to read text on it.
        - After you have the OCR text, do NOT call this tool again for the same question.
        - If the user only wants to know what the drug in the image is, you can answer
        directly from this OCR text without using other tools.

        Input: The user's question about the image (string).
        Output: The raw recognized text from the image (string).
        """

        print(f"\n>> Calling OCR Tool for uploaded image. Query: {query_about_image}")
        
        # Check for a NEW image FIRST.
        if "pending_image_bytes" in st.session_state and st.session_state.pending_image_bytes is not None:
            try:
                image_bytes = st.session_state.pending_image_bytes
                print(f"Processing new image with {len(image_bytes)} bytes...")
                results = reader.readtext(image_bytes)
                recognized_text = " ".join([text for (bbox, text, prob) in results]).strip()

                # Cache the new text
                st.session_state.last_ocr_text = recognized_text or ""
                # CRITICAL: Clear the pending bytes *after* processing
                st.session_state.pending_image_bytes = None

                if not recognized_text:
                    return "Could not extract any text from the new image."
                return f"Recognized text from image: {recognized_text}"
            except Exception as e:
                # Clear the bytes on error to prevent loops
                st.session_state.pending_image_bytes = None
                return f"Error: Could not process new image or extract text. {e}"

        # If no new image, THEN check the cache.
        cached = st.session_state.get("last_ocr_text")
        if cached:
            print("-> Using cached OCR text.")
            return f"Recognized text from image: {cached}"

        # If no new image AND no cache, then there's nothing to process.
        return "Error: I can't find an image to process. Please upload an image first."
        
    @tool
    def rag_search(query: str) -> str:
        """
        Searches a database of drug reviews.
        Use this ONLY for:
        1. Qualitative questions about reviews, side effects, or experiences for a specific drug.
        2. Finding drugs for a specific condition or symptom (e.g., 'drugs for depression').
        Do NOT use for: math, counts, averages, or top lists.
        Input: A search query (string).
        Output: A summary of matching reviews, including snippets and ratings.
        """
        
        print(f"\n>> Calling RAG Tool with query: {query}")
        
        # Pre-process query for symptom-based searches
        processed_query = query
        match = re.search(r"i have (a|an)\s+(.+)", query.lower().strip())
        if match:
            symptom = match.group(2).strip(" .?!")
            processed_query = f"drugs for {symptom}"
            print(f"-> Query re-written to: '{processed_query}'")
        
        # Retrieve documents
        print("-> Using Simple Similarity Retriever...")
        docs = simple_retriever.invoke(processed_query)

        if not docs:
            print("-> Simple Similarity Retriever found 0 documents.")
            return AgentFinish(
                return_values={"output": "No relevant reviews found for that query."},
                log="No relevant reviews found for that query."
            )
        
        print(f"-> Retrieved {len(docs)} documents. Summarizing for agent.")

        summary_lines = [
            f"Found {len(docs)} reviews related to '{processed_query}'. Here is a summary of the retrieved reviews:"
        ]
        
        for i, doc in enumerate(docs):
            drug_name = doc.metadata.get('drugName', 'N/A')
            if drug_name == 'N/A':
                continue 

            full_content = doc.page_content
            review_text = full_content
            
            if 'Review: ' in full_content:
                # Split on 'Review: ' and take the second part [1]
                review_text = full_content.split('Review: ', 1)[1] 
            
            snippet = review_text.split('.')[0].strip() + "."
            rating = doc.metadata.get('rating', 'N/A')
            sentiment = doc.metadata.get('sentiment_label', 'N/A')
            
            summary_lines.append(
                f"{i+1}. Drug: {drug_name.capitalize()} (Rating: {rating}/10, Sentiment: {sentiment}). Snippet: \"{snippet}\""
            )

        if len(summary_lines) <= 1:
             return AgentFinish(
                return_values={"output": "I found reviews, but could not extract their details."},
                log="Could not extract review details."
             )

        final_output = "\n".join(summary_lines)
        return AgentFinish(
            return_values={"output": final_output},
            log=final_output
        )

    @tool
    def get_average_rating(drug_name: str) -> str:
        """
        Calculates the average rating (1-10) for a SINGLE drug.
        Use this for: "average rating", "effectiveness", or "how effective" a drug is.
        Do NOT use for: counts, side effects, or qualitative reviews.
        Input: The exact drug name ONLY (string).
        Output: A sentence with average rating, total review count, and effectiveness.
        """

        print(f"\n>> Calling get_average_rating tool for: {drug_name}")
        # Compute average rating
        try:
            clean_drug_name = drug_name.lower().strip()
            mask = (
                df["drugName"].astype(str).str.strip().str.lower() == clean_drug_name
            )
            filtered = df.loc[mask, "rating"].dropna()
            n = int(filtered.shape[0])
            if n == 0:
                output = f"I'm sorry, I couldn't find any reviews for '{drug_name}' to calculate an average rating."
                return AgentFinish(return_values={"output": output}, log=output)
            
            mean_rating = float(filtered.mean())
            if 1 <= mean_rating <= 4:
                interpretation = "low effectiveness"
            elif 5 <= mean_rating <= 7:
                interpretation = "average effectiveness"
            else:
                interpretation = "high effectiveness"

            output = (
                f"The average rating for {drug_name} (based on {n} reviews) is "
                f"{mean_rating:.1f} out of 10, which indicates {interpretation}."
            )
            return AgentFinish(return_values={"output": output}, log=output)
        except Exception as e:
            output = f"Error calculating rating: {e}"
            return AgentFinish(return_values={"output": output}, log=output)

    @tool
    def get_review_count(drug_name: str) -> str:
        """
        Counts the total number of reviews for a SINGLE drug.
        Use this for: "how many reviews" or "total reviews" for a specific drug.
        Do NOT use for: averages, side effects, or qualitative reviews.
        Input: The exact drug name ONLY (string).
        Output: A sentence with the total review count.
        """

        print(f"\n>> Calling get_review_count tool for: {drug_name}")
        # Compute review count
        try:
            clean_drug_name = drug_name.lower().strip()
            mask = (
                df["drugName"].astype(str).str.strip().str.lower() == clean_drug_name
            )
            n = int(df.loc[mask].shape[0])
            if n == 0:
                output = f"I'm sorry, I couldn't find any reviews for '{drug_name}'."
                return AgentFinish(return_values={"output": output}, log=output)
            output = f"I found a total of {n} reviews for {drug_name}."
            return AgentFinish(return_values={"output": output}, log=output)
        except Exception as e:
            output = f"Error counting reviews: {e}"
            return AgentFinish(return_values={"output": output}, log=output)


    @tool
    def get_top_most(query: str) -> str:
        """
        Finds the Top-5 most reviewed drugs or conditions.
        Use this for: "top 5 drugs", "top 5 conditions", or "top 5 drugs for <condition>".
        Do NOT use for: averages, single-drug counts, or qualitative summaries.
        Input: The user's query, like "top 5 drugs for pain" (string).
        Output: A formatted Top-5 list with counts.
        """
        
        print(f"\n>> Calling get_top_most tool with query: {query}")
        q = (query or "").strip().lower()

        # Intent:
        m_for = re.search(r"top\s*5\s*drugs?\s*for\s+(.+)", q) 
        has_conditions = ("top 5" in q) and ("condition" in q or "conditions" in q) and not m_for
        has_drugs = ("top 5" in q) and ("drug" in q or "drugs" in q) and not m_for

        try:
            cond_col = df["condition"].astype(str).str.strip()
            drug_col = df["drugName"].astype(str).str.strip()

            if m_for:  # top 5 drugs for <condition>
                condition_name = m_for.group(1).strip(" ?.!").strip()
                if not condition_name:
                    return "Please specify a condition, e.g., 'top 5 drugs for Depression'."

                mask = cond_col.str.contains(re.escape(condition_name), case=False, na=False)
                sub = df.loc[mask]
                if sub.empty:
                    return f"I couldn't find any data for the condition: {condition_name}"

                counts = (
                    sub.assign(drug=drug_col.loc[mask])
                    .groupby("drug", dropna=True)
                    .size()
                    .sort_values(ascending=False)
                    .head(5)
                )
                if counts.empty:
                    return f"I couldn't find any drugs for {condition_name}."

                lines = [f"Top 5 most reviewed drugs for {condition_name}:"]
                for i, (name, cnt) in enumerate(counts.items(), 1):
                    lines.append(f"{i}. {name}: {cnt}")
                output = "\n".join(lines)
                return AgentFinish(return_values={"output": output}, log=output)

            if has_conditions:  # top 5 conditions
                counts = cond_col[cond_col.ne("")].value_counts().head(5)
                if counts.empty:
                    return "I couldn't find any conditions."
                lines = ["Top 5 most common conditions:"]
                for i, (name, cnt) in enumerate(counts.items(), 1):
                    lines.append(f"{i}. {name}: {cnt}")
                output = "\n".join(lines)
                return AgentFinish(return_values={"output": output}, log=output)

            if has_drugs:  # top 5 drugs overall
                counts = drug_col[drug_col.ne("")].value_counts().head(5)
                if counts.empty:
                    return "I couldn't find any drugs."
                lines = ["Top 5 most reviewed drugs overall:"]
                for i, (name, cnt) in enumerate(counts.items(), 1):
                    lines.append(f"{i}. {name}: {cnt}")
                output = "\n".join(lines)
                return AgentFinish(return_values={"output": output}, log=output)

            output = "This tool supports: 'top 5 drugs', 'top 5 conditions', or 'top 5 drugs for <condition>'."
            return AgentFinish(return_values={"output": output}, log=output)
        
        except Exception as e:
            output = f"Error finding top list: {e}"
            return AgentFinish(return_values={"output": output}, log=output)
        

    tools = [process_image, rag_search, get_average_rating, get_review_count, get_top_most]

    # --- 5. Create Agent Executor ---
    agent_prompt_template = """
        You are a helpful pharmacy assistant.
        Answer the following questions as best you can. You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        --- RULES (Follow Strictly) ---
        
        **Your first step is to check the user's 'Question' against these rules IN THIS EXACT ORDER:**

        **Rule 1: "Top 5" Questions**
        - If the user's 'Question' asks for a "top 5" list (e.g., "top 5 drugs", "most reviewed conditions"), your action MUST be `get_top_most`.
        - Action Input: The user's full question.
        - **Stop checking other rules.**

        **Rule 2: "Drugs for..." Questions**
        - If the user's 'Question' asks to *find* drugs for a condition or symptom (e.g., "what drugs for a cold?", "what can I take for depression?"), your action MUST be `rag_search`.
        - Action Input: The user's full question.
        - **Stop checking other rules.**

        **Rule 3: Specific Drug Questions**
        - If the user's 'Question' *already contains* a specific drug name (e.g., "what are side effects of Tramadol?", "how effective is Ibuprofen?"), you MUST use a data tool (`rag_search` for experiences/side effects, `get_average_rating` for effectiveness, `get_review_count` for total reviews).
        - **DO NOT use `process_image` if the drug name is already in the 'Question'.**
        - **Stop checking other rules.**

        **Rule 4: Follow-up or Image Questions (Default)**
        - If the 'Question' does *NOT* match Rule 1, 2, or 3 (e.g., it's a follow-up like "how effective is it?" or "what about this one?"), your FIRST and ONLY action MUST be `process_image`.
        - Action Input: The user's full question.
        - After you get the drug name from the 'Observation', you MUST then use a data tool (like `rag_search`) in your next step.

        **NEVER HALLUCINATE (CRITICAL!)**:
           - You MUST ONLY use drug names that are explicitly provided in the 'Question' or a tool's 'Observation'.
           - DO NOT make up drug names or any other information.
           - If the tools don't provide an answer, just say you don't know.

        **PHI / REDACTED TEXT**:
           - The user's query may contain tags like [REDACTED_NAME] or [REDACTED_EMAIL].
           - You MUST NOT try to guess or infer what the redacted text was.
           - Answer the question as if the text was never there.
        
        Begin!
        Question: {input}
        {agent_scratchpad}
        """
    prompt = PromptTemplate.from_template(agent_prompt_template)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Set to True to see detailed logs in console
        handle_parsing_errors=True,
        max_iterations=5
    )

    print("--- Agent Executor created successfully. ---")
    return agent_executor
