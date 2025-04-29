from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory.chat_message_histories import MongoDBChatMessageHistory

class MemoryEntry(BaseModel):
    """Structure for storing game interactions"""
    timestamp: datetime
    entry_type: str  # observation, action, reward, reflection
    content: dict
    metadata: Optional[Dict] = None

class MemoryManager:
    """Manages different types of agent memory"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        mongodb_uri: str,
        agent_id: str,
        collection_prefix: str = "agent_memory"
    ):
        self.llm = llm
        self.agent_id = agent_id
        
        # Working memory for immediate context
        self.working_memory = ConversationBufferMemory(
            memory_key="working_memory",
            return_messages=True,
            k=20
        )

        # Long-term episodic memory in MongoDB
        self.episodic_memory = MongoDBChatMessageHistory(
            connection_string=mongodb_uri,
            session_id=f"{agent_id}_episodic",
            collection_name=f"{collection_prefix}_episodic"
        )

        # Strategic memory for plans and insights
        self.strategic_memory = MongoDBChatMessageHistory(
            connection_string=mongodb_uri,
            session_id=f"{agent_id}_strategic",
            collection_name=f"{collection_prefix}_strategic"
        )

        # Summarization chain
        self.summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a memory curator for an AI agent playing a civilization-style game.
            Summarize the recent game interactions into key strategic insights and lessons learned.
            Focus on:
            1. Important game state changes
            2. Successful and failed strategies
            3. Resource management insights
            4. Enemy behavior patterns
            5. Territory control dynamics"""),
            ("human", "Recent interactions:\n{interactions}\n\nProvide a strategic summary:")
        ])

        self.summarize_chain = LLMChain(
            llm=llm,
            prompt=self.summarize_prompt
        )

    async def add_memory(
        self,
        entry_type: str,
        content: dict,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a new memory entry"""
        entry = MemoryEntry(
            timestamp=datetime.now(),
            entry_type=entry_type,
            content=content,
            metadata=metadata
        )

        # Add to working memory
        self.working_memory.save_context(
            {"entry_type": entry_type},
            {"content": str(content)}
        )

        # Add to episodic memory
        self.episodic_memory.add_message({
            "type": entry_type,
            "content": content,
            "metadata": metadata
        })

        # Check if summarization is needed
        if len(self.working_memory.chat_memory.messages) >= 20:
            await self.summarize_and_store()

    async def summarize_and_store(self) -> None:
        """Summarize recent memories and store strategically"""
        recent_messages = self.working_memory.chat_memory.messages

        # Create summary using LLM
        summary = await self.summarize_chain.arun(
            interactions="\n".join(str(m.content) for m in recent_messages)
        )

        # Store in strategic memory
        self.strategic_memory.add_message({
            "type": "summary",
            "content": summary,
            "timestamp": datetime.now().isoformat()
        })

        # Clear working memory
        self.working_memory.clear()

    async def retrieve_relevant_memories(
        self,
        current_state: dict,
        k: int = 5
    ) -> List[dict]:
        """Retrieve relevant memories based on current state"""
        # Implementation will use similarity search once we add vector store
        # For now, return recent strategic memories
        return self.strategic_memory.messages[-k:]

    async def get_strategic_insights(self) -> str:
        """Get consolidated strategic insights"""
        strategic_memories = self.strategic_memory.messages
        if not strategic_memories:
            return "No strategic insights available yet."
        
        return await self.summarize_chain.arun(
            interactions="\n".join(str(m.content) for m in strategic_memories[-10:])
        )




