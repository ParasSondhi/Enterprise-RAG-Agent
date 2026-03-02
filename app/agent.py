import os
from langchain_community.chat_models import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from app.tools import agent_tools

# 1. Path to your downloaded GGUF model
#MODEL_PATH = os.path.join("models", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")

def build_agent(model_name="llama3.1:8b"):
    print(f"Connecting to Local Ollama Engine: {model_name}...")
    llm = ChatOllama(
        model=model_name, 
        temperature=0
    )
    # 3. The ReAct Prompt (The S-Tier "Thinking" Protocol)
    # This specific template forces the model to think step-by-step
    # 3. The Aggressive ReAct Prompt (Optimized for 1B Models)
   # 3. The Decoy ReAct Prompt (Optimized to prevent plagiarism)
   # 3. The 'Hard Brake' ReAct Prompt
    template = """You are an intelligent enterprise agent. Answer the following questions as best you can. You have access to the following tools:

{tools}

You MUST use the following format STRICTLY:

Question: the input question you must answer
Thought: you should always think about what to do next. (Write your thought here)
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULES FOR LLAMA-3:
1. NEVER put 'Action:' on the same line as 'Thought:'. You MUST write a brief thought, press Enter, and put 'Action:' on a new line.
2. NEVER output the word 'Question:' yourself. You are only generating the Thoughts and Actions.
3. ACCEPT FRAGMENTS: If the tool returns even a tiny piece of relevant information, DO NOT search again. Immediately output 'Thought: I now know the final answer' followed by 'Final Answer: [your summary]'.
4. CONVERSATIONAL BYPASS: If the user says 'hello', 'thanks', or 'ok', DO NOT use a tool. Immediately output 'Thought: This is conversational' followed by 'Final Answer: [your polite response]'.
5. VARY YOUR SEARCHES: If your Observation returns useless headers, title pages, or repeats the same text, DO NOT search for the exact same phrase again. Change your Action Input to be more specific (e.g., instead of 'The Mom Test', search for 'The Mom Test summary' or 'core concepts').
6. STOP SEARCHING IMMEDIATELY: As soon as you see a summary, a list of rules, or an introduction in the Observation, STOP. Do not try to find "better" information. Immediately use what you found to write the Final Answer. Even if the information is short, it is enough.
7. FALLBACK KNOWLEDGE: If the document search does not give you the specific answer you need, DO NOT search again. Output 'Thought: The document lacks specific details, so I will answer using my general knowledge.' and then provide the Final Answer from your own AI training.
8. NEVER, EVER, EVER use the same exact Action Input more than once. If at first you don't succeed, change your search terms immediately. The user is relying on you to be flexible and creative in how you search for information.
9. If you do not found specific information regarding user question, but a vague line then you can use it as a hint to find the answer in your general knowledge.If answer is not found directly but you find answers indirectly related to the question then you can use it as a hint to find the answer in your general knowledge. if you do not find anything in your knowledge then you can even use it as final answer but you must mention that you are not sure about the answer and you are giving your best guess.
Begin!
10. SQL STOP RULE: If the SQL tool returns a raw data array (e.g., [(68.28, 69.31)]), THAT IS YOUR COMPLETE ANSWER. DO NOT search the PDF. DO NOT run the query again. Immediately output 'Thought: I have the SQL data' followed by 'Final Answer:' and format those numbers into a readable English sentence.
11. STRICT COLUMN RULE: You are a precise data scientist. If the user asks for a column that does not exist in the schema (e.g., 'height'), DO NOT substitute it with a similar-sounding column like 'HP'. If the column is missing, your Thought must be 'Column [name] not found in schema' and your Final Answer must be 'I'm sorry, but the dataset does not contain information regarding [column name].'
Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    
    # Inject the SQL context into the prompt permanently
    #prompt = prompt.partial(sql_db_info=sql_description)

    # 4. Bind the Brain, the Tools, and the Prompt together
    print("Binding Tools to the ReAct Agent...")
    agent = create_react_agent(llm, agent_tools, prompt)
    
    # 5. The Executor (The engine that runs the loop)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=agent_tools, 
        verbose=True, 
        handle_parsing_errors=True, # Crucial for local models if they mess up the format
        max_iterations=5, # <--- THE KILL SWITCH (Add this line)
        
    )
    
    return agent_executor

if __name__ == "__main__":
    # 1. Boot the Agent
    agent_executor = build_agent()
    print("\nSUCCESS: Hybrid-Edge ReAct Agent is online.")
    
    # 2. The Interactive Loop
    print("\n--- ENTERING INTERACTIVE MODE ---")
    print("Type 'exit' or 'quit' to shut down the engine.\n")
    
    while True:
        user_input = input("Ask the Database or the Documents >> ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down the engine. Goodbye.")
            break
            
        print("\nWatching the Agent think...\n")
        
        try:
            response = agent_executor.invoke({"input": user_input})
            print("\n==================================")
            print("FINAL AGENT ANSWER:")
            print(response["output"])
            print("==================================")
            print("\n") # Add some breathing room before the next question
        except Exception as e:
            print(f"\nAgent crashed during reasoning: {e}\n")