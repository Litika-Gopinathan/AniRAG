import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Prompt classification
def classify_prompt_type(user_query):
    prompt = f"""You are an expert prompt classifier. Given a user question, classify it into one of these prompt types ONLY:
- System
- Role-Based
- Few-Shot
- Chain of Thought
- Tree of Thought
- ReAct
- Zero-Shot

Return ONLY the prompt type with no other extra words. Do not explain your answer.

Here are some examples:

User Question: "Summarize the rules for interacting with this chatbot."
Output: System

User Question: "As an anime historian, explain the impact of Naruto on modern shonen."
Output: Role-Based

User Question: "Here are three trivia questions about One Piece. What is Luffy's favorite food? Who is the shipwright of the crew? Who has the nickname 'Cat Burglar'?"
Output: Few-Shot

User Question: "What is the plot of Death Note? Think step by step."
Output: Chain of Thought

User Question: "Explain the relationships between all the main characters in Fruits Basket, breaking it down by family and school connections."
Output: Tree of Thought

User Question: "Find the latest episode release date for Jujutsu Kaisen and, if necessary, search online for the answer."
Output: ReAct

User Question: "How many episodes does Bleach have?"
Output: Zero-Shot

User Question: "Give me a list of the main antagonists in Fullmetal Alchemist."
Output: Zero-Shot

User Question: "Act as a veteran anime reviewer and critique the animation style of Demon Slayer."
Output: Role-Based

User Question: "What are the steps Light Yagami took to avoid being caught? Explain in detail."
Output: Chain of Thought

User Question: "Compare the power systems in Hunter x Hunter and My Hero Academia, using a table if helpful."
Output: Few-Shot

User Question: "Break down the tournament arc in Yu Yu Hakusho by rounds and key battles."
Output: Tree of Thought

User Question: "If you don't know the answer, suggest a way to find it using online tools."
Output: ReAct

User Question: "What is the meaning of the word 'nakama' in One Piece?"
Output: Zero-Shot

User Question: "Summarize the chatbot's capabilities and limitations."
Output: System

Now, classify the following user question:

User Question: "{user_query}"
"""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    resp = llm.invoke([HumanMessage(content=prompt)])
    allowed_types = ["System", "Role-Based", "Few-Shot", "Chain of Thought", "Tree of Thought", "ReAct", "Zero-Shot"]
    result = resp.content.strip()
    return result if result in allowed_types else "Zero-Shot"

# Step 2: Anime KB selection using LLM
def get_anime_kb_names():
    return [name.lower() for name in os.listdir("vector_stores") if os.path.isdir(os.path.join("vector_stores", name))]

ANIME_KB_NAMES = get_anime_kb_names()

def select_anime_kb_llm(user_query):
    prompt = f"""You are an expert at identifying which anime knowledge base to use for answering a user's question.
Given a user question, check if it mentions any of the following anime by name (case-insensitive, ignore spaces, underscores, or hyphens):

- berserk
- blackclover
- bleach
- codegeass
- deathnote
- dragonball
- evangelion
- fruitsbasket
- gintama
- haikyuu
- kill-la-kill
- kimetsu-no-yaiba
- madeinabyss
- myheroacademia
- naruto
- onepiece
- onepunchman

Instructions:
- If the user question clearly refers to one or more of these anime, return ONLY the corresponding vectorstore names as a comma-separated list, using the names exactly as listed above (all lowercase).
- If more than one anime is mentioned, return all that are mentioned, separated by commas, in the order they appear in the question.
- If none of these anime are mentioned or you are unsure, return "none".
- Do not explain your answer. Return ONLY the vectorstore name(s) or "none".


Examples:

User Question: "Who is the main villain in Naruto?"
Return: naruto

User Question: "Tell me about the captains in Bleach."
Return: bleach

User Question: "What is the story of Guts in Berserk?"
Return: berserk

User Question: "Compare the power systems in My Hero Academia and Black Clover."
Return: myheroacademia, blackclover

User Question: "What is the meaning of 'nakama' in One Piece?"
Return: onepiece

User Question: "How does Saitama defeat his enemies?"
Return: onepunchman

User Question: "Tell me about the best shonen anime."
Return: none

User Question: "Summarize the plot of Death Note."
Return: deathnote

User Question: "Who are the main characters in Fruits Basket?"
Return: fruitsbasket

User Question: "{user_query}"
"""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    resp = llm.invoke([HumanMessage(content=prompt)])
    anime_names = [name.strip() for name in resp.content.lower().split(",") if name.strip() in ANIME_KB_NAMES]
    return anime_names

# Step 3: On-demand vectorstore loading
def load_vectorstore(anime_name):
    path = os.path.join("vector_stores", anime_name)
    if os.path.isdir(path):
        return FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    return None

# Step 4: Retrieve relevant context (semantic search + metadata)
def retrieve_context(user_query, anime_names, k=4):
    contexts = []
    for anime_name in anime_names:
        vectorstore = load_vectorstore(anime_name)
        if not vectorstore:
            continue
        try:
            # Try filtering by metadata if supported
            docs = vectorstore.similarity_search_with_score(user_query, k=k, filter={"anime": anime_name})
        except Exception:
            # Fallback if metadata filtering is not supported
            docs = [(doc, 1.0) for doc in vectorstore.similarity_search(user_query, k=k)]
        for doc, score in docs:
            if doc.page_content.strip():
                meta = doc.metadata
                title = meta.get("title", "")
                snippet = f"Title: {title}\n{doc.page_content}"
                contexts.append(snippet)
    return "\n\n".join(contexts)

# Step 5: Prompt templates
PROMPT_TEMPLATES = {
    "System": """You are an expert anime assistant. 
Always base your answer strictly and exclusively on the provided context below. 
If the context does not contain enough information to answer the user's question, or if the anime is not present in your knowledge base, respond exactly with: "No data found for this anime or question."

Context:
{context}

User Question:
{user_query}

Your Answer:""",
    "Role-Based": """You are a professional anime character analyst. 
Use only the information in the context below to answer as an expert. 
If the context does not answer the user's question or if the anime is not present, respond exactly with: "No data found for this anime or question." 
Do not use any outside knowledge or make assumptions.

Context:
{context}

User Question:
{user_query}

Expert Analysis:""",
    "Few-Shot": """You are an anime trivia expert. 
Answer the user's question using only the context provided. 
If the answer is not explicitly in the context, or if the anime is not present, respond exactly with: "No data found for this anime or question."
Here are some examples:

Q: Who is the captain of the Straw Hat Pirates?
A: Monkey D. Luffy

Q: {user_query}
A: (Answer concisely using only the context below. If not found, say "No data found for this anime or question.")

Context:
{context}
""",
    "Chain of Thought": """You are an anime plot explainer. 
Think step by step, but use only the information in the context provided. 
If the context does not provide enough information to answer the question, or if the anime is not present, respond exactly with: "No data found for this anime or question." 
Do not invent or assume any details.

Context:
{context}

User Question:
{user_query}

Step-by-step Answer:""",
    "Tree of Thought": """You are an advanced anime analyst. 
Break down the user's question into logical branches or sub-questions, and answer each part using only the context below. 
If any part cannot be answered from the context, or if the anime is not present, respond exactly with: "No data found for this anime or question." 
Do not use any external knowledge or make up information.

Context:
{context}

User Question:
{user_query}

Branch-by-branch Answer:""",
    "ReAct": """You are an anime information assistant. 
Use only the context provided below to answer the user's question. 
If the answer is not present in the context, or if the anime is not present, respond exactly with: "No data found for this anime or question."
If you would normally use a tool or external resource, explain that no data is available.

Context:
{context}

User Question:
{user_query}

Answer (with reasoning or suggested actions if needed):""",
    "Zero-Shot": """You are a precise anime knowledge assistant. 
Answer the user's question using only the context below. 
If the answer is not in the context, or if the anime is not present, respond exactly with: "No data found for this anime or question." 
Do not use outside knowledge or guess.

Context:
{context}

User Question:
{user_query}

Answer:"""
}

def build_prompt(prompt_type, user_query, context):
    template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["Zero-Shot"])
    return template.format(context=context, user_query=user_query)

# Main chatbot pipeline
def chatbot_pipeline(user_query):
    # Step 1: Classify prompt type
    prompt_type = classify_prompt_type(user_query)

    # Step 2: Select anime KB(s)
    anime_names = select_anime_kb_llm(user_query)
    if not anime_names:
        return "No matching anime knowledge base found for your query."

    # Step 3: Retrieve context
    context = retrieve_context(user_query, anime_names)
    if not context.strip():
        return "No data found for this anime or question."
    
    print("[",prompt_type,"]", anime_names)

    # Step 4: Build prompt and get answer
    prompt = build_prompt(prompt_type, user_query, context)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# Example chatbot loop
if __name__ == "__main__":
    print("Anime Chatbot (type 'exit' to quit)")
    while True:
        user_query = input("\nAsk me anything about anime: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        answer = chatbot_pipeline(user_query)
        print("\nAnswer:\n", answer)
