from langchain_groq import ChatGroq
from typing import Dict, List, Any
import json
from datetime import datetime
from model import model  

class ConceptTreeProcessor:
    def __init__(self, concept_tree_path: str = "conceptTree.json", task: str = None):
        """
        Initialize the concept tree processor
        Args:
            concept_tree_path: Path to the concept tree JSON file
            task: Current task being processed (e.g., "festival", "attire")
        """
        self.concept_tree_path = concept_tree_path
        self.concept_tree = self._load_concept_tree()
        self.task = task.lower() if task else None
        
        if self.task and self.task not in self.concept_tree:
            raise ValueError(f"Task '{self.task}' not found in concept tree")
    
    def _load_concept_tree(self) -> Dict:
        """Load the concept tree from JSON file"""
        with open(self.concept_tree_path, 'r') as f:
            return json.load(f)
    
    def _save_concept_tree(self):
        """Save the updated concept tree back to JSON file"""
        with open(self.concept_tree_path, 'w') as f:
            json.dump(self.concept_tree, f, indent=4)
    
    def _categorize_response(self, question: str, answer: str) -> Dict[str, str]:
        """
        Use LLM to categorize the response into relevant concept tree attributes
        """
        messages = [
            ("system", f"""You are an expert at categorizing information about {self.task} into specific attributes.
            Given a question and answer about {self.task}, identify which attributes from the concept tree this information belongs to.
            Return ONLY a JSON object where keys are the matching attribute names and values are the relevant information.
            Only include attributes that are clearly discussed in the response.
            If the information doesn't fit any attributes or is too vague, return an empty JSON object."""),
            ("human", f"""
            Question: {question}
            Answer: {answer}
            
            Available attributes:
            {list(self.concept_tree[self.task].keys())}
            
            Return format:
            {{
                "Attribute Name": "Relevant Information",
                ...
            }}
            """)
        ]
        
        response = model.invoke(messages)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {}
    
    def _merge_information(self, existing_info: str, new_info: str) -> str:
        """
        Use LLM to merge existing information with new information intelligently
        """
        if not existing_info:
            return new_info
            
        messages = [
            ("system", f"""You are an expert at combining and synthesizing information about {self.task}.
            Given existing information and new information about the same {self.task} attribute,
            create a comprehensive and non-redundant combination of both.
            Preserve important details from both sources while eliminating redundancy.
            If the new information contradicts the existing information, prefer the new information
            but mention both perspectives if they might both be valid."""),
            ("human", f"""
            Existing information: {existing_info}
            New information: {new_info}
            
            Provide a merged version that combines both pieces of information effectively. JUST OUTPUT THE INFORMATION AND NOT ANYTHING ELSE""")
        ]
        
        response = model.invoke(messages)
        return response.content
    
    def process_conversation_history(self, conversation_history: List[Dict[str, Any]]):
        """
        Process the conversation history and update the concept tree
        """
        for entry in conversation_history:
            # Categorize the response into relevant attributes
            categorized_info = self._categorize_response(
                entry['question'],
                entry['answer']
            )
            
            # Update each identified attribute
            for attribute, new_info in categorized_info.items():
                if attribute in self.concept_tree[self.task]:
                    existing_info = self.concept_tree[self.task][attribute]
                    merged_info = self._merge_information(existing_info, new_info)
                    self.concept_tree[self.task][attribute] = merged_info
        
        # Save the updated concept tree
        self._save_concept_tree()
    
    def get_missing_attributes(self) -> List[str]:
        """
        Return a list of attributes that still have null values
        """
        return [
            attr for attr, value in self.concept_tree[self.task].items()
            if value is None
        ]
    
    def get_attribute_status(self) -> Dict[str, bool]:
        """
        Return the status of all attributes (filled or not)
        """
        return {
            attr: value is not None
            for attr, value in self.concept_tree[self.task].items()
        }

def process_conversation(conversation_history: List[Dict[str, Any]], task: str):
    """
    Main function to process conversation history and update concept tree
    Args:
        conversation_history: List of conversation entries
        task: Current task being processed (e.g., "festival", "attire")
    """
    processor = ConceptTreeProcessor(task=task)
    processor.process_conversation_history(conversation_history)
    return {
        'missing_attributes': processor.get_missing_attributes(),
        'attribute_status': processor.get_attribute_status()
    }