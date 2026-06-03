import os
import zipfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="Datta Lab RAG", layout="wide")

st.title("Datta Lab RAG Interface")
st.caption("Semantic Knowledge Extraction & Automated Iterative Gap Analysis")

with st.expander("ℹ️ App Instructions & Operational Modes", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### General Inquiry Mode")
        st.write(
            "Designed for standard document lookup. It pulls the top semantic text chunks "
            "matching your prompt and extracts concrete metrics or definitions. If the "
            "context does not contain the answer, it returns an explicit 'I do not know' statement."
        )
    with col2:
        st.markdown("### Research Gap Analysis Mode")
        st.write(
            "Executes an adversarial multi-depth execution loop. Starts with your baseline prompt, "
            "verifies context sufficiency, and if an answer is documented, automatically constructs "
            "exactly 1 deeper technical sub-question probing hardware bounds or mathematical limits. "
            "It breaks and stops drilling immediately once a literature gap is detected."
        )

load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def load_vector_store():
    embeddings = OpenAIEmbeddings()
    persist_directory = './chroma_db'
    zip_path = './chroma_db.zip'
    
    if not os.path.exists(persist_directory) and os.path.exists(zip_path):
        st.text("Unzipping pre-built vector database...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
            
    if os.path.exists(persist_directory):
        st.text("Database found! Loading existing vector store...")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        st.text("No database found. Creating new one (this will cost OpenAI credits)...")
        loader = DirectoryLoader("llamaparse_data/", glob="./**/*.md", loader_cls=TextLoader)
        docs = loader.load()
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="standard_deviation")
        document_chunks = splitter.split_documents(docs)
        db = Chroma.from_documents(documents=document_chunks, embedding=embeddings, persist_directory=persist_directory)
        st.text("Database created and saved!")
        return db

vector_store = load_vector_store()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=open_api_key, temperature=0)
MAX_DEPTH = 3

def get_rag_response(question):
    retrieved_docs = vector_store.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    answer_prompt = f"""
    You are a PHD level research assistant identifying knowledge gaps in scientific literature.
    Use the provided context to answer the question. 
    If the context is insufficient, ambiguous, or does not contain a specific technical answer, 
    you MUST state: "KNOWLEDGE_GAP_DETECTED" and explain what specific information is missing.

    Context:
    {context}

    Question: {question}
    """  

    messages = [
        SystemMessage(content="You are a precise technical researcher."),
        HumanMessage(content=answer_prompt)
    ]

    response = llm.invoke(messages).content
    return response

def generate_deeper_questions(question, answer):
    gen_prompt = f"""
    Based on the previous research question and its current answer, generate exactly 1 highly specific, 
    technical follow-up question that probes deeper into the mathematical, hardware, 
    or implementation constraints of the topic. 
    Focus on finding limits or contradictions.

    Original Question: {question}
    Current Answer: {answer}

    Return only the single question. Do not include any bullet points, numbers, or introduction text.
    """

    messages = [
        SystemMessage(content="Generate granular research questions."),
        HumanMessage(content=gen_prompt)
    ]  

    questions_raw = llm.invoke(messages).content
    return [q.strip() for q in questions_raw.split('\n') if q.strip()]

quest = st.radio("General questions(q) or find gap(g)?", ("General Inquiry Mode (q)", "Find Gap Mode (g)"))

if quest == "Find Gap Mode (g)":
    start_question = st.text_area("Start Question:", height=100)
    if st.button("Start Iterative Research Gap Analysis"):
        if not start_question.strip():
            st.warning("Please enter a starting question.")
        else:
            queue = [(start_question, 0)]
            visited_questions = set()
            identified_gaps = []

            st.markdown("### Execution Progress")
            
            while queue:
                current_question, depth = queue.pop(0)

                if depth > MAX_DEPTH or current_question in visited_questions:
                    continue

                visited_questions.add(current_question)
                
                with st.status(f"Depth {depth} | {current_question[:60]}...", expanded=True) as status:
                    st.write(f"**Question:** {current_question}")
                    answer = get_rag_response(current_question)

                    if "KNOWLEDGE_GAP_DETECTED" in answer:
                        st.error(f"GAP FOUND")
                        st.write(answer)
                        identified_gaps.append({"question": current_question, "gap": answer})
                        status.update(label=f"Depth {depth} | Gap Found", state="error")
                        continue
                    else:
                        st.success("Answer Found")
                        st.write(answer)
                        status.update(label=f"Depth {depth} | Evaluated", state="complete")

                        if depth < MAX_DEPTH:
                            new_questions = generate_deeper_questions(current_question, answer)
                            for nq in new_questions:
                                queue.append((nq, depth + 1))

            st.markdown("### --- Summary of Identified Knowledge Gaps ---")
            if not identified_gaps:
                st.info("No clear gaps identified within the specified depth.")
            else:
                for i, gap in enumerate(identified_gaps):
                    with st.container():
                        st.markdown(f"**{i+1}. Related to:** {gap['question']}")
                        st.write(f"**Detail:** {gap['gap']}")
                        st.markdown("---")

elif quest == "General Inquiry Mode (q)":
    question = st.text_input("Question (type 'q' to clear or stop):")
    if st.button("Submit Question"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif question.strip().lower() == 'q':
            st.info("Inquiry session halted.")
        else:
            with st.spinner("Retrieving document chunks..."):
                retrieved_docs = vector_store.similarity_search(question, k=5)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                answer_prompt = f"""
                You are a PHD level research assistant helping answer researchers questions.
                Use the provided context to answer the question
                If the context is insufficient, ambiguous, or does not contain a specific technical answer, 
                you MUST state: "I do not know" and explain what specific information is missing.

                Context:
                {context}

                Question: {question}
                """  
                messages = [
                    SystemMessage(content="You are a precise technical researcher"),
                    HumanMessage(content=answer_prompt)
                ]

                response = llm.invoke(messages).content
                st.markdown("### Response")
                st.write(response)
