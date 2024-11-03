from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict, ClassVar 
from datetime import datetime
from pydantic import BaseModel, Field
from model import model
import random
from llm2 import ConceptTreeProcessor


tasks = [
    "festival",
]

task = random.choice(tasks)

load_dotenv()


class UserResponse(BaseModel):
    """Model to capture user's response to a question"""
    question: str = Field(description="Question that was asked")
    answer: str = Field(description="User's response to the question")
    timestamp: datetime = Field(default_factory=datetime.now)

class ConversationContext(BaseModel):
    """Model to maintain conversation context and generate relevant follow-up questions"""
    topic: str = Field(description=f"Main topic of conversation (e.g., specific {task})")
    conversation_history: List[UserResponse] = Field(
        description="History of questions and answers",
        default_factory=list
    )
    missing_attributes: List[str] = Field(
        description="List of attributes that need to be collected",
        default_factory=list
    )

class FestivalChatbot(BaseModel):
    f"""Main model for managing {task}-related conversations"""
    
    context: ConversationContext = Field(
        description="Current conversation context"
    )
    
    # Initialize ConceptTreeProcessor for the task
    concept_processor: Optional[ConceptTreeProcessor] = None

    class Config:
        arbitrary_types_allowed = True

    def initialize_concept_processor(self):
        """Initialize the concept tree processor once we know the task"""
        if not self.concept_processor:
            self.concept_processor = ConceptTreeProcessor(task=task)
            self.context.missing_attributes = self.concept_processor.get_missing_attributes()
    
    def generate_next_question(self) -> str:
        """
        Generates the next question using LLM based on conversation history and missing attributes
        """
        if not self.context.conversation_history:
            prompt = f"Initiate a question for this {task}. DO NOT ASK THEM TO START ABOUT ANY PARTICULAR FESTIVAL. ASK THEM ABOUT THEIR FESTIVALS AND NOT A GENERAL QUESTION"
            return model.invoke(prompt).content
        
        # Update concept tree and get missing attributes
        if self.concept_processor:
            self.concept_processor.process_conversation_history([
                {"question": response.question, "answer": response.answer}
                for response in self.context.conversation_history
            ])
            self.context.missing_attributes = self.concept_processor.get_missing_attributes()
        
        # Prepare conversation history for the LLM
        messages = [
            ("system", f"""You are an expert cultural anthropologist chatbot interviewing people about {task}. 
            Generate the next meaningful question based on the conversation history and missing attributes that need to be collected.
            Focus on understanding cultural nuances, traditions, and personal experiences.
            Ask specific, contextual questions that build upon previous responses.
            Prioritize questions about missing attributes but maintain natural conversation flow.
            Don't repeat questions that have already been asked.
            Don't respond with anything except the next question.""")
        ]
        
        # Add conversation history and missing attributes
        conversation_summary = "Conversation history:\n"
        for response in self.context.conversation_history:
            conversation_summary += f"Question: {response.question}\n"
            conversation_summary += f"Answer: {response.answer}\n"
        
        missing_attrs_prompt = "\nMissing attributes that need to be collected:\n"
        missing_attrs_prompt += "\n".join(f"- {attr}" for attr in self.context.missing_attributes)
        
        messages.append(("human", f"{conversation_summary}\n{missing_attrs_prompt}\nBased on this context, what should be the next question? JUST OUTPUT THE QUESTION WHICH CAN BE ANSWERED AND NO OTHER THINGS SHOULD BE PRESNT STRICTLY "))
        
        # Get next question from LLM
        response = model.invoke(messages)
        return response.content
    
    def update_context(self, question: str, answer: str):
        """Updates conversation context with new response"""
        # Add response to history
        self.context.conversation_history.append(
            UserResponse(question=question, answer=answer)
        )
        
        # Update topic if this is the first response
        if len(self.context.conversation_history) == 1:
            self.context.topic = answer
            # Initialize concept processor after we get the first response
            self.initialize_concept_processor()

def create_chatbot():
    return FestivalChatbot(
        context=ConversationContext(
            topic=""
        )
    )

