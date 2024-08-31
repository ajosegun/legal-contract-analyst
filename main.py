from src.config import llm, Config
from src.tools import tools_list
from src.helper import process_uploaded_pdf
from src.prompt_template import prompt_template_contract, prompt_template_chat
from src.guards import validate_unusual_prompt, validate_pii
from src.evaluator import evaluate_res
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser,
)
from langchain.tools import BaseTool
import chainlit as cl
from chainlit.sync import run_sync
from chainlit.types import ThreadDict
from datetime import datetime
from literalai import LiteralClient


literal_client = LiteralClient(api_key=Config.LITERAL_API_KEY)
cb = literal_client.langchain_callback()
today_date = datetime.today().strftime("%d %B %Y")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Will use OAuth or fetch from user DB in real production
    if (username, password) == ("olusegun", "password"):
        return cl.User(
            identifier="Olusegun", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""

        res = run_sync(cl.AskUserMessage(content=query).send())
        return res["content"]

    async def _arun(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        res = await cl.AskUserMessage(content=query).send()
        return res["output"]


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Things that will happen when a chat is resumed.
    """
    chat_message_history = ChatMessageHistory()
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    for message in root_messages:
        if message["type"] == "user_message":
            chat_message_history.add_user_message(message["output"])
        else:
            chat_message_history.add_ai_message(message["output"])
    print(f"chat_message_history on resume: \n\n {chat_message_history.messages}")

    pdf_text = await init_chat()
    # pdf_text = ""
    agent_executor = get_agent_executor(pdf_text)
    cl.user_session.set("agent_executor", agent_executor)
    cl.user_session.set("message_history", chat_message_history)


@cl.on_chat_start
async def start():
    # Initialize the agent

    welcome_message = """## Welcome to the Legal Contract Analysis Assistant!
This assistant specializes in analyzing vendor contracts, comparing them with existing ones, and suggesting improvements based on vendor performance history.

**How to use**

1. You can ask questions about vendor performance, contract analysis, or request suggestions for improvements.
2. The assistant will use its tools to retrieve relevant information and provide insights.

**Examples**

- Analyze the contract for vendor Omega IT Consultants
- What's the performance history of BrightWave Creative?
- Suggest improvements for the EcoEnergy Solutions contract based on their performance

Feel free to ask any questions related to vendor contracts and performance!"""

    await cl.Message(content=welcome_message).send()

    pdf_text = await init_chat()

    chat_message_history = ChatMessageHistory()
    if not cl.user_session.get("message_history"):
        ## first
        cl.user_session.set(
            "message_history",
            chat_message_history,
        )

    agent_executor = get_agent_executor(pdf_text)
    cl.user_session.set("agent_executor", agent_executor)


async def init_chat():
    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
            cl.Action(
                name="Upload a contract",
                value="contract",
                label="✅ Analyse a Contract",
            ),
            cl.Action(name="Chat", value="chat", label="Ask Questions"),
        ],
    ).send()

    pdf_text, pdf_embeddings = "", None
    if res and res.get("value") == "contract":
        files = None

        # Wait for the user to upload a file
        while files is None:
            files = await cl.AskFileMessage(
                content="Please upload a contract PDF file to begin!",
                accept=["application/pdf"],
            ).send()

        if files is not None:
            pdf_file = files[0]
            if pdf_file.path.endswith(".pdf"):
                await cl.Message(
                    content=f"Processing `{pdf_file.name}` for analysis"
                ).send()
                try:
                    pdf_text, pdf_embeddings = process_uploaded_pdf(pdf_file.path)
                    await cl.Message(
                        content=f"Processing complete, you can now analyse `{pdf_file.name}`"
                    ).send()

                except Exception as e:
                    await cl.Message(
                        content=f"Error processing `{pdf_file.name}`. \n {str(e)}"
                    ).send()
                    return
            else:
                await cl.Message(
                    content=f"Unsupported file type `{pdf_file.name}`. Please a PDF file"
                ).send()
                return

    return pdf_text


def get_agent_executor(pdf_text):
    tools = tools_list
    llm_with_tools = llm.bind_tools(tools)
    if pdf_text:
        agent = (
            {
                "input": lambda x: x["input"],
                "contract": lambda x: pdf_text,
                "today_date": lambda x: today_date,
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt_template_contract
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
    else:
        agent = (
            {
                "input": lambda x: x["input"],
                "today_date": lambda x: today_date,
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt_template_chat
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
    )
    return agent_executor


@cl.on_message
async def main(message: cl.Message):
    agent_executor = cl.user_session.get("agent_executor")

    message_history: ChatMessageHistory = cl.user_session.get("message_history")
    message_history.messages = message_history.messages[1:][-7:]
    chat_history = message_history.messages

    answer_prefix_tokens = ["FINAL", "ANSWER"]

    lcb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=answer_prefix_tokens,
    )

    msg = cl.Message(content="")
    await msg.send()

    # validate_unusual_prompt(message.content)
    is_valid, validation_error = validate_pii(message.content)
    if not is_valid:
        msg.content += (
            f"The response contains PII, please try again. \n {validation_error}"
        )
        await msg.update()
        return
    response = await agent_executor.ainvoke(
        {"input": message.content, "chat_history": chat_history},
        config=RunnableConfig(callbacks=[cb, lcb], run_name="analyst"),
    )
    answer = response["output"]

    msg.content += answer
    await msg.update()

    message_history.add_user_message(message.content)
    message_history.add_ai_message(answer)
    cl.user_session.set("message_history", message_history)

    contexts = response.get("intermediate_steps", [])[0][1]

    print(f"contexts: \n {contexts}")
    data = {
        "question": message.content,
        "answer": answer,
        "contexts": contexts,
    }
    # @literal_client.step(type="run", name="ragas")

    @cl.step(type="run", name="ragas")
    def evaluate_response(data_):
        try:
            return evaluate_res(data_)
        except Exception as e:
            print(f"Error evaluating response: {e}")

    evaluate_response(data)
