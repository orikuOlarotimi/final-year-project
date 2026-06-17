import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy
)
from ragas.embeddings.base import embedding_factory
import os
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# 1. MANUAL INPUT SECTION
# -----------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

question = "What are the main steps involved in the research process according to the Research Methodology textbook"

answer = """The main steps involved in the research process according to the Research Methodology textbook are: 1. 
Formulating the research problem: Defining the problem clearly, which may relate to states of nature or relationships 
between variables. 2. Extensive literature survey: Reviewing existing literature relevant to the problem. 3. 
Developing the hypothesis: Formulating hypotheses based on the problem and literature review. 4. Preparing the 
research design: Planning the overall approach and methods for the research. 5. Determining sample design: Deciding 
on the sampling method and size. 6. Collecting the data: Gathering the necessary data for the study. 7. Execution of 
the project: Carrying out the research as per the design. 8. Analysis of data: Processing and analyzing the collected 
data. 9. Hypothesis testing: Testing the hypotheses formulated earlier. 10. Generalizations and interpretation: 
Drawing conclusions and interpreting the results. 11. Preparation of the report or presentation of results: Writing 
up the findings and conclusions formally. This sequence provides a useful procedural guideline for conducting 
research effectively. This information was gotten from pages 27, 29, and 44 of the Research Methodology textbook."""

ground_truth = """The main steps in the research process are: formulating 
the research problem, extensive literature survey, developing the hypothesis, 
preparing the research design, determining sample design, collecting the data, 
execution of the project, analysis of data, hypothesis testing, 
generalisations and interpretation, and preparation of the report."""

contexts = [""" 'constantly anticipating at each step in the research process the requirements of the subsequ
ent\nsteps. However, the following order concerning various steps provides a useful procedural guideline\nr
egarding the research process: (1) formulating the research problem; (2) extensive literature survey;\n(3) 
developing the hypothesis; (4) preparing the research design; (5) determining sample design;\n(6) collectin
g the data; (7) execution of the project; (8) analysis of data; (9) hypothesis testing;', 'Before embarking
 on the details of research methodology and techniques, it seems appropriate to\npresent a brief overview o
f the research process. Research process consists of series of actions or\nsteps necessary to effectively c
arry out research and the desired sequencing of these steps. The\nchart shown in Figure 1.1 well illustrate
s a research process.\n10 Carlos L. Lastrucci, The Scientific Approach: Basic Principles of the Scientific 
Method, p. 7.', '(10) generalisations and interpretation, and (11) preparation of the report or presentatio
n of the results,\ni.e., formal write-up of conclusions reached.\nA brief description of the above stated s
teps will be helpful.\n1. Formulating the research problem:  There are two types of research problems, viz.
, those\nwhich relate to states of nature and those which relate to relationships between variables. At the
', 'government and other agencies doing this job in our country. Researcher also faces the\nproblem on acco
unt of the fact that the published data vary quite significantly because of\ndifferences in coverage by the
 concerning agencies.\n10. There may, at times, take place the problem of conceptualization  and also probl
ems\nrelating to the process of data collection and related things.\nQuestions\n1. Briefly describe the dif
ferent steps involved in a research process.', 'the following steps generally one after the other: (i) stat
ement of the problem in a general way; (ii)\nunderstanding the nature of the problem; (iii) surveying the a
vailable literature (iv) developing the\nideas through discussions; and (v) rephrasing the research problem
 into a working proposition.\nA brief description of all these points will be helpful.\n(i) Statement of the problem in a general way: First of all the problem should be stated in a' """]

# -----------------------------
# 2. EVALUATION LOGIC
# -----------------------------
async def run_evaluation():
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=client
    )
    # instantiate all four metrics
    faithfulness_scorer = Faithfulness(llm=llm)
    context_precision_scorer = ContextPrecision(llm=llm)
    context_recall_scorer = ContextRecall(llm=llm)
    answer_relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
    # run all four at the same time
    results = await asyncio.gather(
        faithfulness_scorer.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        ),
        context_precision_scorer.ascore(
            user_input=question,
            reference=ground_truth,
            retrieved_contexts=contexts
        ),
        context_recall_scorer.ascore(
            user_input=question,
            reference=ground_truth,
            retrieved_contexts=contexts
        ),
        answer_relevancy_scorer.ascore(
            user_input=question,
            response=answer,
        ),
    )

    faithfulness, precision,  recall, answer_relevancy = results

    print("\n========== RAGAS RESULTS ==========")
    print(f"Faithfulness:       {faithfulness.value:.4f}")
    print(f"Context Precision:  {precision.value:.4f}")
    print(f"Context Recall:     {recall.value:.4f}")
    print(f"Answer Relevancy:   {answer_relevancy.value:.4f}")

# -----------------------------
# 3. RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    asyncio.run(run_evaluation())
