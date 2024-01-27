## [See demo here](https://docs.google.com/presentation/d/11txNfT8DS9v0zBsQ6qoua9-BaAquALhURHtiCf9rZDo/edit?usp=sharing)

 An AI + ML tool that both detects potential and technical failures in space missions, and also assists in designing them.

# High-level project summary:

STAR is a precision-engineered AI tool intricately tailored for comprehensive space mission validation. Employing a logistic regression model fine-tuned to a 97% accuracy, it analyzes over 50 space missions, capturing their success metrics, design nuances, and outcomes. This data primes and trains STAR's language learning model to identify technical discrepancies, requirement inconsistencies, and points of failure for potential missions. As a result, STAR delivers targeted feedback and actionable changes, equipping mission planners with insightful feedback and optimized guidelines, all derived from empirical mission data, ensuring enhanced reliability and success probability for upcoming space ventures.


# Detailed project solution

### How does it work?

S.T.A.R. (Spacecraft Technical Analysis Resource) is a synergy of machine learning and Natural Language Processing (NLP) specifically tailored for rigorous space mission validation. It meticulously dissects mission objectives and parameters. The wealth of knowledge from its expansive database of over 50 prior space missions empowers STAR to align new objectives with historical insights, illuminating patterns, correlations, and potential risks.

Its logistic regression algorithm fine-tuned to an accuracy of 97%, is specifically crafted to detect incongruities within mission designs. By contrasting the current mission specifics against a rich tapestry of historical data and industry benchmarks, STAR identifies both technical misalignments and potential vulnerabilities. But STAR's capabilities aren't merely diagnostic; it proactively offers actionable remediations.

Drilling down, STAR's machine learning backbone meticulously parses the successes and setbacks from an array of space missions, distilling key features that shaped their outcomes. Parallelly, its advanced language model doesn't just interpret; it innovates. Continually refined by historical data, this model proactively crafts mission designs, ensuring alignment with proven methodologies while avoiding known hazards. This harmonization of analytic rigor with generative prowess positions STAR as an indispensable ally in space mission strategy, blending retrospective insights with prospective innovation.

The processing pipeline is as follows:

Data Preparation: Initial steps include crawling the dataset, handling missing values, optimizing data structure, and recalibrating features, such as converting 'Year' to 'Mission_Age' to set the age of a mission relative to the present year. <br>

Feature Engineering: Textual data from columns like 'Objective', 'Outcome', and 'Reason' are transformed into numerical vectors via the TF-IDF methodology.<br>

Model Training: The data undergoes a 70-30 train-test split. The logistic regression model, once trained on this dataset, is then assessed using the test set.<br>

Performance Evaluation: STAR employs learning curves and 10-fold cross-validation to ensure model robustness and gauge its performance with varying data volumes.<br>

Guideline Generation for LLM: STAR categorizes historical missions by their outcomes, subsequently crafting mission guidelines anchored in empirical success and failure patterns.<br>

Collaborative filtering is a crucial asset, enabling STAR to draw analogies between upcoming and past missions, spotlighting erstwhile challenges and suggesting optimized trajectories. STAR’s language model, primed by logistic regression insights, acts as a conduit, liaising with users and offering sagacious counsel. It positions STAR as a robust mission consultant, juxtaposing novel space ventures against a chronicle of mission insights, ensuring that planners are equipped with resilient and refined blueprints.

### What do you hope to achieve?

By integrating machine learning and language models, STAR seeks to improve the design, planning, and validation processes of space missions. To streamline the process of sifting through technical requirements, detecting omissions, inconsistencies, and offering requirement recommendations. Additionally, by analyzing over 50 past missions, STAR aims to pinpoint common failures and successes and reduce risk, allowing future missions to avoid known pitfalls and build upon proven strategies. 

Furthermore, by identifying potential issues early in the planning phase, the project hopes to minimize expensive last-minute adjustments or mission failures, leading to more cost-effective space missions. Consequently, the language learning model within STAR is geared towards crafting novel mission designs that not only learn from the past but also innovate for the future. By refining and curating existing mission plans based on historical data and advanced modeling, STAR aims to ensure that space missions evolve in efficiency, safety, and success rates. Equally as knowledge consolidation, STAR serves as a reservoir of insights from decades of space missions, providing a platform where historical knowledge meets cutting-edge technology to guide future space endeavors.

In essence, the STAR project aspires to revolutionize space mission planning and validation, ensuring that future missions are more successful, efficient, and innovative by leveraging the lessons of the past and the capabilities of advanced AI technologies.

### Tools, Software, Hardware

**Software and Libraries**:

• Python: The main programming language.

• pandas: Used for data manipulation and analysis.

• seaborn: For statistical data visualization.

• matplotlib: For MATLAB-like plotting and data visualization.

• numpy: For numerical computations. To support arrays (including multidimensional arrays), as well as assortments of mathematical functions to operate on these arrays.

• openai: For language learning model and API calls.

• langchain: <br>
ConversationalRetrievalChain, RetrievalQA: Used for retrieval in conversational settings.
<br>
ChatOpenAI: Used for chat-based interfaces with OpenAI models.
<br>
DirectoryLoader, TextLoader: Load data/documents from directories or text sources.
<br>
OpenAIEmbeddings: Represents embeddings from the OpenAI library.
<br>
VectorstoreIndexCreator: For creating an index for Vectorstore.
<br>
VectorStoreIndexWrapper: A wrapper for the Vectorstore index.
<br>
OpenAI: An interface or module related to OpenAI models in the context of langchain.
<br>
Chroma: Related to the vectorstores in langchain.
<br>
PromptTemplate: Template to create prompts for models.
<br>
LLMChain, SequentialChain: Chain interfaces for LLM (Language Learning Model) and sequential operations.
<br>
ConversationBufferMemory: Memory module for storing conversational buffers.

• scikit-learn: <br>
ColumnTransformer: Transforms features by applying transformers to parts of the input.
<br>
Pipeline: Used to assemble several steps that can be cross-validated together.
<br>
OneHotEncoder: Used for encoding categorical features as a one-hot numeric array.
<br>
StandardScaler: Used to standardize the dataset's features.
<br>
train_test_split: Used to split arrays or matrices into random train and test subsets.
<br>
TfidfVectorizer: Converts a collection of raw documents to a matrix of TF-IDF features.
<br>
LogisticRegression: Represents a logistic regression classifier.
<br>
classification_report: Used to build a text report showing the main classification metrics.
<br>
cross_val_score: Evaluates a score by cross-validation.
<br>
learning_curve: Determines cross-validated training and test scores for different training set sizes.
<br>

### References
**Space Agency References**: <br>
https://www.nasa.gov/missions <br>
https://pds.nasa.gov/ <br>
https://data.nasa.gov/dataset/Space-Shuttle-Main-Propulsion-System-Anomaly-Detec/hwtc-f6bz <br>
https://mars.nasa.gov/ <br>
https://nssdc.gsfc.nasa.gov/nmc/ <br>
https://nssdc.gsfc.nasa.gov/nmc/SpacecraftQuery.jsp <br>


### Tags

#ai #artificialintelligence #machinelearning #design #spacemissions #naturallanguageprocessing #optimization
