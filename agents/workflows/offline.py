from typing import Dict, List
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI  # Updated import
from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import Graph

from ..core.memory import MemoryManager

class OfflineProcessor:
    """Handles offline processing tasks like memory consolidation"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        memory_manager: MemoryManager,
        summarization_interval: timedelta = timedelta(hours=1)
    ):
        self.llm = llm
        self.memory = memory_manager
        self.summarization_interval = summarization_interval
        self.last_summarization = datetime.now()
        
        # Chain for generating strategic insights
        self.insight_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an strategic advisor analyzing game history.
            Review the recent game experiences and extract key strategic insights.
            Focus on:
            1. Successful patterns and strategies
            2. Common failure modes
            3. Resource management principles
            4. Territory control tactics
            5. Long-term planning insights
            
            Provide structured insights that can guide future game episodes."""),
            ("human", """Recent game history:
            {history}
            
            Previous insights:
            {prev_insights}
            
            Generate updated strategic insights:""")
        ])
        
        # self.insight_chain = LLMChain(
        #     llm=llm,
        #     prompt=self.insight_prompt
        # )
        self.insight_chain = self.insight_prompt | llm | StrOutputParser()
        
    async def create_offline_workflow(self) -> Graph:
        """Create workflow for offline processing"""
        
        workflow = Graph()
        
        # Add nodes for different offline processing steps
        workflow.add_node("check_summarization", self._check_summarization_needed)
        workflow.add_node("get_recent_memories", self._get_recent_memories)
        workflow.add_node("summarize_memories", self._summarize_memories)
        workflow.add_node("update_strategic_knowledge", self._update_strategic_knowledge)
        
        # Define edges
        workflow.add_edge("check_summarization", "get_recent_memories")
        workflow.add_edge("get_recent_memories", "summarize_memories")
        workflow.add_edge("summarize_memories", "update_strategic_knowledge")
        
        # Set entry point
        workflow.set_entry_point("check_summarization")
        
        return workflow.compile()
    
    async def _check_summarization_needed(self, state: Dict) -> Dict:
        """Check if it's time for memory summarization"""
        current_time = datetime.now()
        if current_time - self.last_summarization >= self.summarization_interval:
            state["should_summarize"] = True
            self.last_summarization = current_time
        else:
            state["should_summarize"] = False
        return state
    
    async def _get_recent_memories(self, state: Dict) -> Dict:
        """Retrieve recent memories for processing"""
        if not state.get("should_summarize", False):
            return state
        
        # Get memories since last summarization
        recent_memories = await self.memory.episodic_memory.messages
        if recent_memories:
            state["recent_memories"] = recent_memories
        return state
    
    async def _summarize_memories(self, state: Dict) -> Dict:
        """Generate summary of recent experiences"""
        if not state.get("should_summarize", False) or "recent_memories" not in state:
            return state
            
        memories = state["recent_memories"]
        summary = await self.memory.summarize_and_store()
        state["memory_summary"] = summary
        return state
    
    async def _update_strategic_knowledge(self, state: Dict) -> Dict:
        """Update long-term strategic knowledge"""
        if not state.get("should_summarize", False) or "memory_summary" not in state:
            return state
            
        # Get previous insights
        prev_insights = await self.memory.get_strategic_insights()
        
        # Generate new insights
        new_insights = await self.insight_chain.ainvoke(
            history=state["memory_summary"],
            prev_insights=prev_insights
        )
        
        # Store updated insights
        await self.memory.strategic_memory.add_message({
            "type": "strategic_insights",
            "content": new_insights,
            "timestamp": datetime.now().isoformat()
        })
        
        return state