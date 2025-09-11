import gspread
from google.oauth2.service_account import Credentials
import os
import json
from pymilvus import MilvusClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# --- Setup for Google Sheets ---
SERVICE_ACCOUNT_FILE = r'C:\Users\hamza\Documents\Heriot-Watt\HWU-AI-Learning-Buddy-Copilot\notebooks\learning-buddy-099-22b47deab465.json'
SHEET_URL = r'https://docs.google.com/spreadsheets/d/16g0I-lYROjaUkeVcUoCOKOyh_CwN9Zyz2QUS2S8aY_E/edit?gid=1932960332#gid=1932960332'
WORKSHEET_NAME = 'F21CA'

try:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_url(SHEET_URL).worksheet(WORKSHEET_NAME)
    print("Successfully connected to Google Sheet.")
except Exception as e:
    print(f"Error connecting to Google Sheet: {e}")
    exit()

# --- Setup connections from your app.py script ---
# These are configured to connect to your local Ollama and Milvus instances
milvus_uri = "http://localhost:19530"
client = MilvusClient(uri=milvus_uri)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm = Ollama(
    model="llama3",
    request_timeout=300.0,
)

# Your collection names based on the app.py script
COURSE_COLLECTIONS = {
    "F21CA": "HWU_MACS_F21CA",
    "F21NL": "HWU_MACS_F21NL",
}

def rewrite_query(original_query: str, chat_history: list):
    """
    Rewrites a user's query using the LLM for better retrieval,
    based on the current and historical chat messages.
    """
    history_text = "\n".join([f"{m.role}: {m.content}" for m in chat_history])
    prompt = (
        "Given the following conversation history and a new user query, rewrite the user query to "
        "be a standalone, comprehensive query that is more effective for semantic search. "
        "Do not include any chat history in your final answer. "
        "Provide only the rewritten query, nothing else.\n\n"
        f"Chat history:\n{history_text}\n\n"
        f"User query: {original_query}\n\n"
        "Rewritten query:"
    )
    
    response = llm.complete(prompt)
    return response.text.strip()

def get_rag_response(query: str, course_id: str = WORKSHEET_NAME, chat_history: list = []):
    """
    Sends a query to the RAG system and returns the response.
    Replicates the core logic from app.py.
    """
    try:
        current_collection = COURSE_COLLECTIONS.get(course_id)
        if not current_collection:
            return "Course not found."
        
        # Rewrite the query for better context retrieval
        # rewritten_query = rewrite_query(query, chat_history) NOT USED
        
        # Generate a query embedding
        query_embedding = embed_model.get_query_embedding(query)
        
        # Retrieve context from Milvus
        results = client.search(
            collection_name=current_collection,
            data=[query_embedding],
            anns_field="embedding",  # REQUIRED
            search_params={'metric_type': 'COSINE', 'params': {'nprobe': 10}},  # REQUIRED
            output_fields=["context"],
            limit=5,
        )

        context_chunks = [hit['entity']['context'] for hit in results[0]]
        full_context = "\n".join(context_chunks)

        # Build the system prompt
        system_prompt = ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are a helpful and approachable course assistant for HWU students. "
                "Your goal is to answer questions using ONLY the provided CONTEXT. "
                "This CONTEXT is in Markdown format. "
                "First, identify the key FACTS from the CONTEXT that directly address the user's query. "
                "Then, use those FACTS to construct your final answer. "
                "If the CONTEXT does not contain enough information to answer, respond with: 'I don’t know based on the available course information.' \n"
                "Do not generate advice, instructions, or help unrelated to the retrieved context."
                "Do not assist with assignments, essays, reports, quizzes, or courseworks"
                "Keep answers concise and factual.\n\n"
                f"Course: {course_id}\n"
                f"Original query: {query}\n"
                # f"Rewritten query: {rewritten_query}\n\n"
                f"CONTEXT:\n{full_context}"
            )
        )

        # Reconstruct full history for LLM
        full_chat_history = [system_prompt] + chat_history

        # Pass full conversation history to LLM
        response = llm.chat(full_chat_history)
        return response.message.content

    except Exception as e:
        return f"ERROR: Failed to get response. {e}"

def main(course_id: str = WORKSHEET_NAME):
    """Reads questions, gets agent responses, and writes them back to the sheet."""
    try:
        data = worksheet.get_all_records()
        print(f"Loaded {len(data)} rows from the sheet.")
        
        # Get column indices
        headers = worksheet.row_values(1)
        try:
            q_col = headers.index('Questions')
            fu_q_col = headers.index('Follow-up Question(s)')
            resp_col = headers.index('Agent Response')
            fu_resp_col = headers.index('Agent Follow-up response')
            
        except ValueError as e:
            print(f"Error: Missing required column in the sheet. Make sure columns 'Questions', 'Follow-up Question(s)', 'Agent Response', and 'Agent Follow-up response' exist. {e}")
            return
            
        # Iterate through each row of the data
        for i, row in enumerate(data):
            try:
                question = row['Questions']
                follow_up_question = row['Follow-up Question(s)']
                current_agent_response = row['Agent Response']
                
                # Skip the row if a response has already been generated
                if current_agent_response and current_agent_response.strip() != '':
                    print(f"Skipping row {i+2}. 'Agent Response' column is already filled.")
                    continue

                print(f"Processing row {i+2}: Question = '{question}' for course '{course_id}'")

                # Get the initial response
                chat_history = []
                agent_response = get_rag_response(question, course_id, chat_history)

                # If there's a follow-up, get the next response
                if follow_up_question and follow_up_question.strip().upper() != 'N/A':
                    # Add the initial user prompt and agent response to the chat history
                    chat_history.append(ChatMessage(role=MessageRole.USER, content=question))
                    chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=agent_response))
                    
                    follow_up_response = get_rag_response(follow_up_question, course_id, chat_history)
                else:
                    follow_up_response = "N/A"

                # Update the sheet with the results
                worksheet.update_cell(i + 2, resp_col + 1, agent_response)
                worksheet.update_cell(i + 2, fu_resp_col + 1, follow_up_response)
                print(f"-> Successfully updated row {i+2}")
            
            except Exception as e:
                print(f"An error occurred while processing row {i+2}: {e}")
                continue

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    main()