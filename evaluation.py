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
Formulating the research problem 2. Extensive literature survey 3. Developing the hypothesis 4. Preparing the 
research design 5. Determining sample design 6. Collecting the data 7. Execution of the project 8. Analysis of data 
9. Hypothesis testing 10. Generalisations and interpretation 11. Preparation of the report or presentation of the 
results (formal write-up of conclusions reached) These steps provide a useful procedural guideline for effectively 
carrying out research, with an emphasis on anticipating the requirements of subsequent steps throughout the process. 
Additionally, Fig. 1.1 on page 28 illustrates this research process as a flowchart showing the sequential steps and 
feedback loops, highlighting the iterative and controlled nature of research."""

ground_truth = """The main steps in the research process are: formulating 
the research problem, extensive literature survey, developing the hypothesis, 
preparing the research design, determining sample design, collecting the data, 
execution of the project, analysis of data, hypothesis testing, 
generalisations and interpretation, and preparation of the report."""

contexts = ["""constantly anticipating at each step in the research process the requirements of the subsequent\nsteps. However, the following order concerning various steps provides a useful procedural guideline\nregarding the r
esearch process: (1) formulating the research problem; (2) extensive literature survey;\n(3) developing the hypothesis; (4) preparing the research design; (5) determining sample design;\n(6) collecting the data; (7) execution of
 the project; (8) analysis of data; (9) hypothesis testing;', 'Before embarking on the details of research methodology and techniques, it seems appropriate to\npresent a brief overview of the research process. Research process c
onsists of series of actions or\nsteps necessary to effectively carry out research and the desired sequencing of these steps. The\nchart shown in Figure 1.1 well illustrates a research process.\n10 Carlos L. Lastrucci, The Scien
tific Approach: Basic Principles of the Scientific Method, p. 7.', "[Figure on page 28 — Fig. 1.1]: A horizontal flowchart titled 'RESEARCH PROCESS IN FLOW CHART' showing the sequential steps in a research process within a large
 light blue rectangular background. The flowchart consists of seven main rectangular nodes connected with rightward arrows, each representing a research stage labeled with Roman numerals I through VII below each node. The steps 
are as follows: I - 'Define research problem'; from this step, two downward dashed arrows lead respectively to 'Review concepts and theories' and 'Review previous research finding' inside two smaller dashed rectangles arranged v
ertically on the right side. Both review steps have rightward arrows that converge and lead to step III - 'Formulate hypotheses'. Step III connects rightward to IV - 'Design research (including sample design)', which then procee
ds rightward to V - 'Collect data (Execution)'. Next, an arrow leads rightward to VI - 'Analyse data (Test hypotheses if any)', followed by step VII - 'Interpret and report'. Two feedback loops marked with blue circles containin
g 'F' connect from step VI back to V and from V back to IV respectively. Two feed forward loops marked with blue circles containing 'FF' proceed from the top middle and right top corner of the large rectangle back to steps IV an
d the main flow start respectively, showing the flow feedback mechanisms. At bottom right below the flowchart, a legend explains 'F' is feedback, helping in controlling the sub-system, and 'FF' is feed forward, serving the vital
 function of providing criteria for evaluation. The figure effectively illustrates the iterative and controlled nature of the research process.", '(10) generalisations and interpretation, and (11) preparation of the report or pr
esentation of the results,\ni.e., formal write-up of conclusions reached.\nA brief description of the above stated steps will be helpful.\n1. Formulating the research problem: There are two types of research problems, viz., thos
e\nwhich relate to states of nature and those which relate to relationships between variables. At the', 'government and other agencies doing this job in our country. Researcher also faces the\nproblem on account of the fact that
 the published data vary quite significantly because of\ndifferences in coverage by the concerning agencies.\n10. There may, at times, take place the problem of conceptualization and also problems\nrelating to the process of data collection and related things.\nQuestions\n1. Briefly describe the different steps involved in a research process."""]

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
