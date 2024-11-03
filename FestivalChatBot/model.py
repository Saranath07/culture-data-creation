from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()


model = ChatGroq(
    temperature=0.7,
    model="llama-3.1-70b-versatile",
)


