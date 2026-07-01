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

question = "types of hypertension"

answer = """The document lists these types of hypertension: - **Essential (primary) hypertension**: over 90% of 
cases; no identifiable cause is found. - **Secondary hypertension**: less than 10% of cases; caused by an 
identifiable underlying condition. Additionally, it says secondary hypertension can be caused by things like renal 
disease, renovascular problems, vascular causes, hormone/steroid intake, adrenal disorders, other endocrine 
disorders, neurogenic causes, prohypertensive substances, pregnancy, hypercalcemia, and sleep apnea. this information 
was gotten from page 6"""

ground_truth = """ Essential (primary) hypertension:  
That represents over 90% of the cases. It is diagnosed in the absence of an identifiable cause. 
However, many risk factors increase the likelihood of its occurrence (Table 1): and Secondary hypertension: 
That represents less than 10% of all cases. It is defined as hypertension resulting from an 
underlying identifiable cause (Table 2).  """

contexts = ["""  'hypertension in adults aged 18 years or older is: \n \nSystolic blood pressure ≥140 mmHg and/or diastolic blood pressure ≥90 mmHg on the \naverage of two or more readings taken at each of two or more visits after 
initial screening. \n \nTypes of hypertension: \n \nEssential (primary) hypertension:  \nThat represents over 90% of the cases. It is diagnosed in the absence of an identifiable cause. \nHowever, many risk factors increase the l
ikelihood of its occurrence (Table 1): \n \nTable 1', '6 \n \nSedentary life style \nStress \nFamily history of hypertension, age, gender and race (non modifiable risk) \n \n \nSecondary hypertension: \nThat represents less than
 10% of all cases. It is defined as hypertension resulting from an \nunderlying identifiable cause (Table 2). \n \nTable 2 \nIdentifiable causes of secondary hypertension \n \nSource or category  of cause \nPossible causes \n \n
Renal diseases \nRenal parenchymal disease \nPolycystic kidney \nUrinary tract obstruction \nRennin-producing tumor', '7 \n \nThe diagnosis is made clinically by a high BP and rapidly progressive end organ damage \nsuch as retin
opathy (grade 3 or 4), renal dysfunction (especially proteinuria) and/or \nhypertensive encephalopathy.  If left untreated, death will occur within few months. \n \n \nClassification of hypertension: \n \nTable 3  \nClassificati
on of blood pressure for adults* \n Blood pressure class \nSystolic blood \npressure (mmHg) \nDiastolic blood \npressure (mmHg) \nNormal \n<120 \nAnd <80 \nPrehypertension', '[Table on page 6]:\n| Source or category of cause | P
ossible causes |\n| --- | --- |\n| Renal diseases | Renal parenchymal disease\nPolycystic kidney\nUrinary tract obstruction\nRennin-producing tumor\nLiddle syndrome |\n| Renovascular hypertension | Renal artery stenosis\nConnect
ive tissue disease\nGlomerulonephritis |\n| Vascular | Coarctation of aorta\nVasculitis/polycythemia\nCollagen vascular disease |\n| Hormone and steroid intake | Oral contraceptives\nEstrogen replacement therapy\nOral and Depot 
contraceptives,\nSteroid medication |\n| Adrenal | Primary aldosteronism\nCushing syndrome\nPheochromocytoma\nCongenital adrenal hyperplasia |\n| Other endocrine disorders | Hyperthyroidism and hypothyroidism\nHyperparathyroidis
m\nAcromegaly |\n| Neurogenic | Brain tumor\nLesions of brainstem or hypothalamus\nRaised intracranial pressure |\n| Prohypertensive substances | Adrenergic medication, nasal\ndecongestants\nNonsteroidal anti-inflammatory drugs\
nAnti-depressants (tricyclic, MAOI),\nAlcohol, Cyclosporine and Tacrolimus,\nerythropoietin |\n| Other | Pregnancy\nHypercalcemia\nSleep Apnea |', 'ARB, BB, CCB) as \nneeded \nStage 2 \nHypertension \n≥160 \nor 100 \nYes \nTwo-d
rug \ncombination for \nmost (usually \nthiazide-type diuretic \nand ACEI or ARB or \nBB or CCB) \n \n* Compelling indication: target organ damage or associated clinical condition or risk factors \n \nLifestyle Modifications: \n \nThe adoption of healthy lifestyles by all persons is critical for the prevention of hypertension \nand is an indispensable part of the management of those with hypertension (Annex 3).'"""]

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
