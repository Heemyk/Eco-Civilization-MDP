from typing import Dict, List
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

class StrategicPlan(BaseModel):
    """Strategic plan for agent actions"""
    primary_objective: str
    sub_objectives: List[str]
    resource_priorities: Dict[str, float]
    expansion_targets: List[Dict[str, int]]  # List of coordinate targets
    threat_assessments: List[Dict]
    opportunity_scores: Dict[str, float]

class StrategicPlanner:
    """Generates strategic plans based on observations and memory"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        self.planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strategic planner for a civilization-style game agent.
            Based on the current game state analysis and historical insights, create a strategic plan that:
            1. Sets clear primary and secondary objectives
            2. Prioritizes resource acquisition and usage
            3. Identifies expansion opportunities
            4. Assesses threats and opportunities
            5. Balances short-term gains with long-term development
            
            Structure your response to populate a StrategicPlan object."""),
            ("human", """Current state analysis: {state_analysis}
            Historical insights: {historical_insights}
            Available resources: {available_resources}
            
            Generate a strategic plan:""")
        ])
        
        self.planning_chain = LLMChain(
            llm=llm,
            prompt=self.planning_prompt
        )
    
    async def create_plan(
        self,
        state_analysis: str,
        historical_insights: str,
        available_resources: Dict
    ) -> StrategicPlan:
        """Generate a strategic plan based on current state and history"""
        
        # Get raw plan from LLM
        plan_text = await self.planning_chain.arun(
            state_analysis=state_analysis,
            historical_insights=historical_insights,
            available_resources=str(available_resources)
        )
        
        # Parse and structure the plan
        try:
            # Process the LLM output into structured format
            lines = plan_text.strip().split("\n")
            plan_dict = {}
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.endswith(":"):
                    current_section = line[:-1].lower().replace(" ", "_")
                    plan_dict[current_section] = []
                elif line and current_section:
                    if current_section == "resource_priorities":
                        resource, priority = line.split(":")
                        if "resource_priorities" not in plan_dict:
                            plan_dict["resource_priorities"] = {}
                        plan_dict["resource_priorities"][resource.strip()] = float(priority)
                    else:
                        plan_dict[current_section].append(line)
            
            # Create StrategicPlan object
            return StrategicPlan(
                primary_objective=plan_dict.get("primary_objective", [""])[0],
                sub_objectives=plan_dict.get("sub_objectives", []),
                resource_priorities=plan_dict.get("resource_priorities", {}),
                expansion_targets=self._parse_coordinates(plan_dict.get("expansion_targets", [])),
                threat_assessments=self._parse_threats(plan_dict.get("threats", [])),
                opportunity_scores=self._parse_opportunities(plan_dict.get("opportunities", []))
            )
            
        except Exception as e:
            # Fallback to a basic plan if parsing fails
            return StrategicPlan(
                primary_objective="Survive and expand",
                sub_objectives=["Gather resources", "Explore territory", "Build infrastructure"],
                resource_priorities={"food": 1.0, "production": 0.8, "gold": 0.6},
                expansion_targets=[],
                threat_assessments=[],
                opportunity_scores={"exploration": 0.8, "development": 0.7}
            )
    
    def _parse_coordinates(self, coord_strings: List[str]) -> List[Dict[str, int]]:
        """Parse coordinate strings into structured format"""
        coords = []
        for s in coord_strings:
            try:
                x, y = map(int, s.replace("(", "").replace(")", "").split(","))
                coords.append({"x": x, "y": y})
            except:
                continue
        return coords
    
    def _parse_threats(self, threat_strings: List[str]) -> List[Dict]:
        """Parse threat strings into structured format"""
        threats = []
        for s in threat_strings:
            if ":" in s:
                source, level = s.split(":")
                threats.append({
                    "source": source.strip(),
                    "threat_level": float(level.strip())
                })
        return threats
    
    def _parse_opportunities(self, opp_strings: List[str]) -> Dict[str, float]:
        """Parse opportunity strings into scores"""
        opportunities = {}
        for s in opp_strings:
            if ":" in s:
                opp, score = s.split(":")
                opportunities[opp.strip()] = float(score.strip())
        return opportunities