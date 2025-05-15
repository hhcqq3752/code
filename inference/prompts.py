ATTRIBUTE_LABELING_PROMPT = """
You will be shown a pairwise preference data point consisting of the following three components:

- `conversation_history`: A string containing a single-turn or multi-turn dialogue between a user and an assistant.
- `candidate_1`: A string containing the assistant's response to the conversation.
- `candidate_2`: A different string containing another assistant's response to the conversation.

Your task is to assess the context and responses, and extract the following five high-level annotation attributes: **task category**, **preference objectivity**, **controversiality**, **desired attributes**, and **annotation guideline**. These attributes help improve response quality by enabling fine-grained analysis of preferences. Please follow the instructions and definitions below carefully.

---

### **Attribute Definitions**

1. **Task Category**: Select one category from the list below that best describes the main task or intent in the conversation. Choose **exactly one** from:

   - Advice seeking
   - Brainstorming
   - Coding & Debugging
   - Creative writing
   - Data analysis
   - Editing
   - Information seeking
   - Math
   - Planning
   - Reasoning
   - Role playing
   - Other
   
   You should only choose "Other" when none of the above fit. Use it carefully.

2. **Preference Objectivity**: Decide whether, based on the conversation context, the responses reflect **objective** or **subjective** preferences:

   - **Objective**: The better response is determined by clear, logical, factual, or task-specific standards. Most humans (or LLMs) would likely agree.
   - **Subjective**: The better response depends on personal taste, tone, or context. Different users (or LLMs) might reasonably disagree.

3. **Controversiality**: Estimate the likelihood that different humans (or LLMs) might **disagree** on which response is better:

   - **Low**: Consensus is likely; disagreement is rare.
   - **Medium**: Some disagreement is possible, but not dominant.
   - **High**: Strong potential for disagreement due to personal, stylistic, or sensitive factors.

4. **Desired Attributes**: List the high-level qualities that a good response should demonstrate in this context. These may be general (e.g., "clarity", "depth") or context-specific (e.g., "empathy", "vivid imagery"). These attributes should reflect what users are likely to value in a high-quality answer.

5. **Annotation Guideline**: Write a concise, context-specific instruction to guide a human annotator in determining which response is better. This should:

   - Reflect the task type, objectivity level, controversiality, and desired attributes,
   - Reduce human annotator's cognitive load by clarifying key decision criteria (i.e., design a method to evaluate the responses),
   - Include additional context, such as the objectivity level or task-specific nuances,
   - Optionally specify needed external knowledge or tools (e.g., search engine, external database, or a specialized LLM), if judging the responses requires it.

---

### **Response format**

Return **valid JSON** with exactly the following five keys in this order:

```json
{{
  "task_category": "<one of: Advice seeking | Brainstorming | Coding & Debugging | Creative writing | Data analysis | Editing | Information seeking | Math | Planning | Reasoning | Role playing | Other>",
  "preference_objectivity": "<one of: Objective | Subjective>",
  "controversiality": "<one of: Low | Medium | High>",
  "desired_attributes": "<comma-separated list of key qualities>",
  "annotation_guideline": "<a complete guideline that would help a human annotator evaluate which of the two responses is better>"
}}
```

Do **not** include any additional keys, comments, or explanatory text—only the JSON object.

Some examples are provided below:

```json
{{
  "task_category": "Coding & Debugging",
  "preference_objectivity": "Objective",
  "controversiality": "Low",
  "desired_attributes": "correctness, clarity, best practices, completeness",
  "annotation_guideline": "Choose the response that provides a functionally correct solution to the user's coding problem. Verify that the code runs as intended based on the user's described issue. Prefer the candidate that includes accurate syntax, proper handling of edge cases, and follows standard best practices (e.g., variable naming, error handling). If both are correct, favor the more readable and maintainable version with concise comments or explanations."
}}
```

```json
{{
  "task_category": "Advice seeking",
  "preference_objectivity": "Subjective",
  "controversiality": "Medium",
  "desired_attributes": "empathy, specificity, relevance, practicality",
  "annotation_guideline": "Select the response that offers advice most directly applicable to the user's situation. Prioritize responses that explicitly acknowledge the user's concerns, avoid generic platitudes, and give specific, actionable suggestions. Emotional tone matters—prefer responses that show empathy without sounding overly scripted. Minor factual errors are less important than helpfulness and emotional alignment."
}}
```

```json
{{
  "task_category": "Information seeking",
  "preference_objectivity": "Objective",
  "controversiality": "Low",
  "desired_attributes": "factual accuracy, relevance, clarity, conciseness",
  "annotation_guideline": "Pick the response that provides the most accurate and relevant information based on the user's query. Cross-check named facts (e.g., dates, statistics, definitions) against reputable sources like Wikipedia or Britannica if unsure. Disregard responses with factual errors, vague answers, or off-topic content. Among accurate responses, prefer the one that explains concisely and clearly without unnecessary elaboration."
}}
```

```json
{{
  "task_category": "Creative writing",
  "preference_objectivity": "Subjective",
  "controversiality": "Medium",
  "desired_attributes": "originality, vivid imagery, narrative coherence",
  "annotation_guideline": "Select the response that is more compelling as a piece of creative writing. Look for originality in ideas, vivid and sensory-rich language, and a coherent narrative arc. If both responses are imaginative, favor the one with more stylistic flair, emotional depth, or unique voice. Avoid choosing based on personal genre preference—evaluate based on execution."
}}
```

```json
{{
  "task_category": "Reasoning",
  "preference_objectivity": "Objective",
  "controversiality": "Low",
  "desired_attributes": "logical consistency, clarity, structured explanation",
  "annotation_guideline": "Choose the response that presents a logically sound and well-structured argument. Check whether the reasoning is consistent with the user's question, free of fallacies, and supported by evidence or explanation. Prefer responses that break down steps clearly (e.g., using numbered or ordered logic), avoid vague assertions, and draw correct conclusions from premises."
}}
```

```json
{{
  "task_category": "Planning",
  "preference_objectivity": "Objective",
  "controversiality": "Medium",
  "desired_attributes": "feasibility, detail, organization, relevance",
  "annotation_guideline": "Pick the response that provides a more feasible and detailed plan tailored to the user's goal. Prioritize structured responses that include clear steps, timelines, or dependencies. Eliminate responses that are overly generic, missing key steps, or suggest impractical or irrelevant actions. Between well-scoped plans, prefer the more organized and actionable one."
}}
```

---

### **Input**

[START OF CONVERSATION HISTORY]
{conversation_history}
[END OF CONVERSATION HISTORY]

[START OF CANDIDATE 1 RESPONSE]
{candidate_1}
[END OF CANDIDATE 1 RESPONSE]

[START OF CANDIDATE 2 RESPONSE]
{candidate_2}
[END OF CANDIDATE 2 RESPONSE]

Please make sure your output is consistent with the definitions above and is **strictly formatted as a valid JSON object** with the required keys and values. Return **only** the JSON object.
""".strip()

ANNOTATION_PROMPT = """
# Preference Data Annotation Protocol

This protocol establishes a standardized method for annotating preference pairs for training reward models. Each annotation task involves evaluating a conversation history along with two candidate responses to identify which response better meets the user's needs, adheres to the user's instructions, or satisfies general desired attributes within the context of the conversation.

## General Annotation Guidelines

### Annotation Process

1. Read the conversation history and the two candidate responses.
2. Read the desired attributes and annotation guidelines for the task.
3. Select the better response based on the desired attributes, annotation guidelines, or external knowledge if applicable.

### Core Principles

Successful annotation relies on four core principles:

- **Objectivity:** Base decisions on defined criteria rather than the annotator's personal preferences, especially for those with objective measures of quality. Even for subjective tasks, the provided desired attributes and annotation guidelines should be used as the criteria for annotation instead of the annotator's personal stance.
- **Verification:** When applicable, ensure assessments are grounded in accurate information and consistent standards.
- **Conservativeness:** As long as the given two responses are of similar quality, or the provided information is insufficient to make a decision, mark the task as "no preference".

### Prohibited Practices

To maintain annotation quality, avoid these practices:

- Do not rely primarily on personal preferences when annotating objective tasks that have clear correctness criteria.
- Do not prioritize style over substance when the content's correctness is paramount.

## Category-Specific Guidelines

The following task-specific guidelines are provided as reference, which you can choose to follow or not.

### Brainstorming

For brainstorming tasks, rely on human judgment. Key criteria include the quantity and diversity of ideas, relevance to the user's request, originality and creativity, and the practicality of suggestions.

Count the number of distinct, relevant ideas. Assess how well these ideas align with the specific brainstorming request. Evaluate their originality. Prioritize responses that provide structured organization of ideas to enhance usability.

### Coding & Debugging

Evaluate based on functional correctness (whether the code solves the stated problem), error handling and edge case management, efficiency and optimization, and adherence to readability standards and best practices.

Check for potential security issues or vulnerabilities. Assess code readability by examining variable naming, comments, and overall structure. Prefer solutions that explain the approach alongside the code.

### Creative Writing

Evaluate based on adherence to requested creative parameters, narrative coherence and structure, stylistic quality and language use, and originality.

First evaluate how well the response fulfills the specific creative request. Assess the narrative structure and coherence. Examine stylistic elements like imagery, voice, and tone. Judge originality independent of personal genre preferences.

### Data Analysis

Key criteria include correctness of analysis methods, appropriateness of statistical approaches, clarity in presenting insights, and accuracy of calculations.

Confirm that the statistical methods are appropriate. Evaluate how clearly the response explains its insights. Assess any visualization recommendations for appropriateness and clarity.

### Editing

Evaluate based on grammar and syntax correctness, clarity improvements, retention of original meaning, and stylistic enhancement when specifically requested.

Compare the original text with the edited version to identify improvements. Check for grammar and syntax corrections. Verify that the original meaning has been preserved or enhanced. Assess any stylistic changes against the specific editing request.

### Information Seeking

Evaluate responses based on factual accuracy, relevance to the query, comprehensiveness, and clarity of explanation.

Identify key factual claims in each response. Assess how completely the response addresses the information request. Check carefully for factual errors or misleading information that could misinform the user.

### Math

Key criteria include mathematical correctness, clear step-by-step reasoning, appropriateness of the solution approach, and completeness in addressing all parts of the problem.

Verify each solution step independently. Check final answers through separate calculations. Confirm that all aspects of the problem have been addressed. Prefer responses that include clear explanations rather than just numerical answers.

### Planning

Evaluate based on the feasibility of the proposed plan, comprehensiveness of included steps, clarity of timeline or sequencing, and consideration of relevant constraints.

Evaluate the practicality of the proposed plan given the context. Check for any missing critical steps. Assess the organization and clarity of the planning sequence. Consider how effectively the plan addresses any constraints mentioned by the user.

### Reasoning

Key criteria include logical consistency, soundness of argument, clarity and structure of explanation, and validity of conclusions drawn from premises.

Break down the reasoning into distinct logical steps. Verify each logical connection individually for soundness. Check carefully for fallacies or invalid inferences.

### Role Playing

Evaluate based on adherence to the specified character or role, consistency with the established universe or setting, appropriateness of tone and language, and engagement with the user's scenario.

Evaluate character consistency throughout the response. Check factual accuracy within the fictional universe when applicable. Assess how effectively the response engages with the user's specific scenario. Consider whether the tone and language are appropriate for the specified role.

### Advice Seeking

Key criteria include relevance to the specific situation, practicality and actionability of suggestions, empathy and understanding, and safety and responsibility.

Assess how directly the advice addresses the user's specific situation. Check the practicality of any suggested actions. Evaluate how well the response demonstrates empathetic understanding. Verify that the advice is responsible and unlikely to cause harm.

### Objectivity / Subjectivity

#### Objective Tasks

For tasks with objective criteria like math, coding, information seeking, and logical reasoning, identify verifiable elements such as facts, calculations, or code functionality. Base your decision primarily on the final result rather than presentation style. If both responses are objectively correct (or incorrect), mark the task as "no preference".

#### Subjective Tasks

For subjective tasks like creative writing, advice, and role playing, apply category-specific criteria consistently across responses. Consider alignment with the user's intent and contextual needs.

#### Mixed Tasks

Many tasks contain both objective and subjective components. Verify objective elements first to establish a factual baseline.

### Controversiality Guidelines

#### Low Controversiality

Apply the standard verification process with a focus on correctness and desired attributes. These topics typically have clear factual bases or widely accepted standards.

#### Medium Controversiality

Focus your preference on responses that offer balanced, well-reasoned approaches. Prioritize responses that acknowledge complexity or present multiple perspectives.

#### High Controversiality

Focus your preference on responses that acknowledge the controversy directly, present balanced viewpoints, avoid unsubstantiated claims, and clearly distinguish between factual statements and opinions.

### Human-Labeled Examples

{human_examples}

### Example to Label

[START OF CONVERSATION HISTORY]
{conversation_history}
[END OF CONVERSATION HISTORY]

[START OF CANDIDATE 1 RESPONSE]
{candidate_1}
[END OF CANDIDATE 1 RESPONSE]

[START OF CANDIDATE 2 RESPONSE]
{candidate_2}
[END OF CANDIDATE 2 RESPONSE]

Please first analyze the human-labeled examples and the example to label. Then, make your best judgment on which response is better. Make sure your response is consistent with the guidelines above and uses the JSON format.
""".strip()