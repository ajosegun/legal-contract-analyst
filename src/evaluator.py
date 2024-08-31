from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.integrations.langchain import EvaluatorChain

# from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas import evaluate
from src.config import llm, embeddings_model, Config
from literalai import LiteralClient

literal_client = LiteralClient(api_key=Config.LITERAL_API_KEY)

# list of metrics we're going to use
metrics = [
    faithfulness,
    answer_relevancy,
    # context_recall, ## need ground_truth to calculate
    # context_precision,
    # answer_correctness,
]

# make eval chains
eval_chains = {
    m.name: EvaluatorChain(metric=m, llm=llm, embeddings=embeddings_model)
    for m in metrics
}

# create evaluation chains
# faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
# answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
# context_rel_chain = RagasEvaluatorChain(metric=context_precision)
# context_recall_chain = RagasEvaluatorChain(metric=context_recall)


# @literal_client.step(type="run", name="ragas")
def evaluate_res(result):
    # evaluate
    # print(f"Evaluating response: {result}")
    # print(f"Evaluating response: {eval_chains.keys()}")
    scores_dict = {}
    for name, eval_chain in eval_chains.items():
        try:
            print(f"Evaluating response for : {name}")
            eval_result = eval_chain(result)
            print(f"eval_chain(result): {eval_result.keys()}")
            score_name = f"{name}_score"
            print(f"{score_name}: {eval_result.get(name, '')}")
            scores_dict[score_name] = eval_result.get(name, "")
        except Exception as e:
            print(f"Error evaluating response: {e}")

    return scores_dict


# result = evaluate(
#     amnesty_qa["eval"], metrics=metrics, llm=llm, embeddings=embeddings_model
# )
