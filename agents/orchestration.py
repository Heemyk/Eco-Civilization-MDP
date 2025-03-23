from typing import List, Dict
import asyncio
from actor import LLMActor
from critic import TripletCritic
from base import *

class GameOrchestrator:
    def __init__(
        self,
        env,
        actors: Dict[str, LLMActor],
        critic: TripletCritic,
        max_iterations: int = 1000
    ):
        self.env = env
        self.actors = actors
        self.critic = critic
        self.max_iterations = max_iterations

    async def run_episode(self):
        iteration = 0
        while iteration < self.max_iterations:
            # Get global state
            global_state = self.env.get_global_state()
            
            # Get suggestions for each actor
            suggestions = {}
            for agent_id, actor in self.actors.items():
                suggestion = await self.critic.generate_suggestion(
                    global_state,
                    agent_id
                )
                suggestions[agent_id] = suggestion
            
            # Get consensus from actors
            consensus = await self._get_consensus(suggestions)
            
            if consensus:
                # Execute actions
                rewards = await self._execute_actions(suggestions)
                # Update memories
                await self._update_memories(suggestions, rewards)
            else:
                # Generate feedback and get new suggestions
                await self._handle_disagreement(suggestions, global_state)
            
            iteration += 1

    async def _get_consensus(
        self, 
        suggestions: Dict[str, Action]
    ) -> bool:
        # Collect all actors' evaluations
        evaluations = await asyncio.gather(*[
            actor.evaluate_suggestion(
                self.env.observe(agent_id),
                suggestions[agent_id]
            )
            for agent_id, actor in self.actors.items()
        ])
        
        # Check if all actors agree
        return all(eval[0] for eval in evaluations)
