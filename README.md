# STATS201_Yiwei_Final_Project
This repository is the final project for STATS201.The project applies the Alpaca-LoRA method to fine-tune large language models to analyze the sentiment of social media tewwts.
## Project information
- Author: Yiwei Liang, Computation and Design with tracks in Computer Science, Class of 2025, Duke Kunshan University
- Instructor: Prof. Luyao Zhang, Duke Kunshan University
- Acknowledgments: I extend my deepest gratitude to Prof. Luyao Zhang for their unwavering guidance, expertise, and encouragement, which helped me finish this project.
- Disclaimer: Submissions to the Final Project for [STATS201 Introduction to Machine Learning for Social Science, 2023 Autumn Term (Seven Week - Second)](https://ms.pubpub.org/) instructed by Prof. Luyao Zhang at Duke Kunshan University.
- Project summary: This project deploys Alpaca-LoRA to fine-tune the LlaMA2 model for analyzing social media sentiments. It labels tweets, preprocesses it, and integrates sentiment analysis. Alpaca-LoRA optimizes the LlaMA2 model, enhancing its relevance in sentiment analysis. LlaMA2 undergoes fine-tuning, incorporating insights from the adapted model for precise sentiment analysis. This approach refines large language models, enabling a new approach to the analysis of sentiment of social media tweets from news data, and aiding informed social studies.
## Poster and Project Summary
### Background
- This project explores the utilization of the Alpaca-LoRA method to fine-tune large language models for the purpose of analyzing sentiments within social media content. The focus is on adapting the LlaMA2 model, a collection of pre-trained and fine-tuned generative text models, to enhance its capabilities in interpreting sentiment tendencies prevalent in discussions across various platforms.
### Research question
Can sentiment in social media replies be accurately predicted using large language models fine-tuned by LoRA (Low-Rank Adaptation of Large Language Models) and relevant datasets?
### Application Scenario
- This project integrating Alpaca-LoRA to fine-tune the LlaMA2 model revolutionizes how journalists decipher social media sentiment. As breaking news erupts, the system swiftly gathers tweets, meticulously labeling and preprocessing them for analysis. Alpaca-LoRA steps in, optimizing the LlaMA2 model to grasp the nuances of social media discourse, amplifying its relevance in sentiment analysis. The model undergoes a meticulous fine-tuning process, absorbing key insights from the adapted framework, honing its precision in gauging sentiments expressed across tweets. This cutting-edge methodology not only refines large language models but reshapes the approach to dissecting social media sentiments related to news events. Journalists now access a tool that swiftly decodes public reactions, aiding in crafting more resonant stories. Moreover, researchers find a goldmine for social studies, unraveling the intricate tapestry of societal responses to news through these refined sentiment analyses.
### Methodology
- Data Collection: Gather diverse labeled social media tweets.
- Preprocessing: Extract relevant text, remove noise, correct errors, and convert data to the required format.
- Alpaca-LoRA Integration: Integrate Alpaca-LoRA for optimal large language model integration, fine-tuning it for enhanced relevance in gold price analysis.
- LlaMA2 Fine-Tuning: Enhance LlaMA2 for gold price prediction using insights from the adapted large language model. Train the model to predict gold price movements based on news content analysis.
- Model Training and Evaluation: Train the adapted LlaMA2 model with preprocessed data and insights. Evaluate its performance using metrics like accuracy, and precision.
### Results
- Insights Generation: Interpret and present quantitative and qualitative insights derived from the model's predictions, showcasing the effectiveness of utilizing large language models for analyzing social media tweets sentiments.
- ![](Result\1.png)
- ![](Result\2.png)
- ![](Result\3.png)
- ![](Result\4.png)
### Intellectual Merits and Practical impacts
- The intellectual merits of this project lie in its pioneering approach to refining sentiment analysis through the integration of Alpaca-LoRA and the LlaMA2 model. By deploying cutting-edge technology to fine-tune established language models, it advances the understanding of social media sentiments in the context of news data. The project not only enhances the relevance and accuracy of sentiment analysis but also contributes to the evolution of large language models, paving the way for more nuanced and contextually aware AI systems.
- Practically, this endeavor offers substantial impacts across various domains. In media and journalism, it facilitates real-time insights into public sentiment, enabling faster and more precise news reporting. Businesses benefit from more informed decision-making through the understanding of consumer opinions and trends. Furthermore, in academia, this refined approach aids researchers in comprehensively studying societal responses to events, enriching social studies and fostering a deeper understanding of human behavior in the digital age.
### Poster
- ![](STATS201.png)
## Literature
- Connecting it to prior research, the paper by Neethu, M. S., and R. Rajasree (2013) focuses on sentiment analysis in Twitter using machine learning techniques. It likely provides foundational insights into the challenges and methods for sentiment analysis in the context of Twitter data. Meanwhile, Agarwal and Mittal's work in 2016 on machine learning approaches for sentiment analysis, especially the prominent feature extraction for sentiment analysis, could offer techniques and methodologies related to feature extraction, a crucial aspect in sentiment analysis models.
- This advances this prior research in several ways:
1. Model Optimization: The utilization of Alpaca-LoRA to fine-tune the LlaMA2 model represents a novel approach. This optimization process enhances the LlaMA2 model's relevance and performance in sentiment analysis, potentially addressing limitations or challenges identified in earlier models.
2. Improved Accuracy: Fine-tuning LlaMA2 by integrating insights from the adapted model improves the precision of sentiment analysis on social media tweets. This advancement likely tackles issues like context understanding, sarcasm detection, or handling informal language prevalent in tweets, thus contributing to more accurate sentiment analysis.
3. New Methodology: By refining large language models through this optimization process, the research introduces a new methodology or approach to sentiment analysis on social media data, especially tweets from news sources. This could pave the way for a more robust and efficient sentiment analysis framework applicable to various social studies.
4. Addressing Contemporary Needs: Given that the cited papers were published earlier, this current research addresses contemporary needs by likely considering newer challenges in sentiment analysis, such as evolving language usage, diverse sentiments, and the dynamic nature of social media conversations.
- In general, this project builds upon prior research by implementing advanced techniques (Alpaca-LoRA) to enhance an existing model (LlaMA2) for sentiment analysis. It not only contributes to improving accuracy but also introduces a refined methodology that could potentially influence future research in social media sentiment analysis and its applications in social studies.
### Flowchart
- ![](Literature\1.png)
## Method
### 1.1. The Prediction Problem
#### Research Question Formulation:
- Objective: Can sentiment in social media replies be accurately predicted using large language models fine-tuned by LoRA (Low-Rank Adaptation of Large Language Models) and relevant datasets?
- Significance: Understanding sentiment in social media replies holds crucial importance for various applications such as gauging public opinion, brand perception, and trend analysis. Accurate sentiment analysis aids in informed decision-making for businesses, governments, and organizations by comprehending public sentiment towards specific topics or events.
#### Operational Measures:
##### Variables:
- Dependent Variable (Y): Sentiment of social media replies (positive, negative, neutral).
- Independent Variables (X): Features extracted from text data including word embeddings, context, and linguistic patterns.
Data Type: The dataset is likely to be cross-sectional, comprising social media replies sampled at a specific time.
### Hypothesis Development:
- Prediction Hypothesis: The linguistic features and context within social media replies (X variables) can accurately predict sentiment (Y variable).
- Justification: Natural language inherently contains patterns and contextual cues that reflect sentiment. Extracting these features using models like LoRA enables the prediction of sentiment with a reasonable level of accuracy due to its capacity to capture contextual information within language data.
### Machine Learning Algorithm Selection:
- Algorithm: LoRA or similar models that capture the low-rank structure of language data are suitable. LoRA specializes in adapting large language models to specific domains, making it well-suited for social media sentiment analysis.
Justification: LoRA's capability to adapt to specific datasets and effectively capture contextual information within language aligns with the intricate nature of social media text data.
### 1.2.The Machine Learning Workflow
#### Model Development:
##### Data Processing: 
###### Methodology:The data processing involves several steps to quantify the variables of interest. This includes:
- Tokenization: Breaking down text into individual tokens (words, phrases, etc.).
- Cleaning: Removing noise, irrelevant characters, symbols, and standardizing text (lowercasing, stemming, lemmatization).
- Feature Extraction: Generating word embeddings (word2vec, GloVe), extracting syntactic and semantic features, capturing linguistic patterns, and contextual information from social media replies.
#### Results Presentation:
- Training and Testing: The results will be presented by employing a train-test split, such as a 70/30 ratio. The model will be trained on a subset (70%) of the data and evaluated on unseen test data (30%) to assess its performance and generalization capability.
- Data Visualization Approach: Visualization techniques like confusion matrices, ROC curves, and precision-recall curves will be utilized to effectively communicate key insights derived from the model's predictions and performance metrics.
#### Model Evaluation:
##### Evaluation Criteria: The model's performance will be assessed using various metrics such as:
- Accuracy: Overall correctness of the predictions.
- Precision: Proportion of correctly predicted positive cases among the total predicted positive cases.
- Recall: Proportion of correctly predicted positive cases among the actual positive cases.
##### Iterative Improvement Strategies:
- Hyperparameter Tuning: Adjusting model parameters to optimize performance (learning rate, regularization, etc.).
Feature Engineering: Exploring additional features or refining existing ones to enhance predictive power.
- Ensemble Methods: Combining multiple models to improve accuracy or robustness.
### Flowchart 
![](method\1.png)
## Data
### Download data
- Download from https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset
### Description
- The dataset contains several columns: 'textID' identifying each entry, 'text' representing the original text, 'selected_text' indicating a selected part of the original text, and 'sentiment' classifying the sentiment as positive, negative, or neutral. The 'text' column contains various tweets, while 'selected_text' represents a snippet of the original text that carries the sentiment. Sentiments are diverse, ranging from positive (like "fun" and "interesting"), neutral (like URLs or general statements), to negative (such as "bullying" or expressions of sadness and frustration). The dataset seems derived from social media or text-based platforms, with sentiments extracted to train sentiment analysis or similar models. The 'selected_text' could be the focal point for sentiment analysis algorithms, teaching them to identify sentiment-bearing phrases within larger texts.
### Data Dictionary
|  **Variable** 	|             **Definition**            	|                          **Description**                          	|   **Type**  	|                                                  **Sample Observations**                                                 	|
|:-------------:	|:-------------------------------------:	|:-----------------------------------------------------------------:	|:-----------:	|:------------------------------------------------------------------------------------------------------------------------:	|
|     textID    	| Unique identifier for each text entry 	|          Alphanumeric string identifying each text entry          	| Categorical 	|                                            cb774db0d1, 549e992a42, 088c60f138,                                           	|
|      text     	|             Original text             	|               Textual content of the original tweet               	|     Text    	| "I`d have responded, if I were going," "Sooo SAD I will miss you here in San Diego!!!", "my boss is bullying me...", ... 	|
| selected_text 	|   Selected part of the original text  	| Extracted snippet of text reflecting the sentiment or key content 	|     Text    	|                           "I`d have responded, if I were going", "Sooo SAD", "bullying me", ...                          	|
|   sentiment   	|         Sentiment of the text         	|  Categorized sentiment of the text (positive, negative, neutral)  	| Categorical 	|                                                neutral, negative, positive                                               	|
#### Frequency/Ranges/Units:

- textID: Unique alphanumeric strings.
- text and selected_text: Variable-length textual data.
- sentiment: Categorical variable with three categories (positive, negative, neutral).
#### Type:
- textID: Categorical (Alphanumeric)
- text and selected_text: Text (String)
- sentiment: Categorical (String)
#### Sample Observations:
- textID: cb774db0d1, 549e992a42, 088c60f138, ...
- text: "I`d have responded, if I were going," "Sooo SAD I will miss you here in San Diego!!!", "my boss is bullying me...", ...
- selected_text: "I`d have responded, if I were going", "Sooo SAD", "bullying me", ...
- sentiment: neutral, negative, positive, ...
### Flowchart
![](Data\1.png)
## Code
### Description
- The data query process involves importing a dataset using Pandas in Python, reading a CSV file, and creating a JSON file. The code initializes an empty list for data storage, iterates through each row of the dataset, and constructs a structured output based on specific columns. This structured data is formatted into a dictionary and appended to the list. Finally, the list containing the structured data is written to a JSON file using the json.dump method. This process is language-agnostic and showcases the general steps of reading, processing, and exporting data, adaptable to languages like Java and Go with their respective syntax adjustments.
### pseudo-code
- ![](2.png)
```
\begin{algorithm}
    \caption{Data Query}
  \begin{algorithmic}
    \REQUIRE pandas, data csv
    \INPUT data csv
    \OUTPUT Json file for data processing
    \STATE \textbf{Create} Empty array $dataset\_data=[]$
    \STATE \textbf{Create Intermediate Data} Integer 
 $temp\_data1=\text{length of data}$
    \FOR{$0 \leq i\leq temp\_data1$, $i++$}
      \STATE \textbf{Create Intermediate Data} String $temp\_data2$
      \STATE \textbf{Add Content} $temp\_data2$ add $data[i]$
      \STATE \textbf{Construct Final Dataset} $dataset\_data=[]$ append $data\_format\_1$
      \STATE \textbf{The format for above} $data\_format\_1=$
        \STATE\{\"instruction": "Detect the sentiment.",
        \STATE"input": text of the data[i],
        \STATE"output": $temp\_data2$\}
        \ENDFOR
        \STATE \textbf{Import Json Relate Packages}
        \STATE \textbf{Save file as json}
  \end{algorithmic}
\end{algorithm}
```
### Flowchart
- ![](Code\1.png)
## Results:
- This research aims to answer the question of accurately predicting sentiment in social media replies using large language mohdels fine-tuned by LoRA and relevant datasets through a comprehensive methodology:
- Dataset Selection and Preprocessing: We meticulously select and preprocess diverse social media datasets containing reply interactions. These datasets encompass a wide array of topics, ensuring a comprehensive representation of social media discourse.
- LoRA Integration and Fine-Tuning: Leveraging LoRA, we fine-tune large language models such as LlaMA2 with the selected datasets. LoRA's low-rank adaptation techniques enable the models to grasp the nuanced sentiment expressions within social media replies.
- Model Evaluation and Validation: Employing rigorous evaluation metrics, we validate the fine-tuned models' performance in predicting sentiment accurately within social media replies. Cross-validation and benchmarking against established sentiment analysis methods ensure robustness and reliability. 
- Result Analysis and Interpretation: Analyzing the outcomes, we detail the effectiveness of LoRA-fine-tuned models in predicting sentiment. We delve into specific cases, showcasing instances where the models excel and elucidating areas for potential improvement.
- Discussion and Conclusion: The research culminates in a discussion emphasizing how the amalgamation of LoRA adaptation techniques with large language models significantly enhances sentiment prediction in social media replies. We highlight the practical implications, limitations, and avenues for future research in refining sentiment analysis methodologies.
- ![](Result\5.png)
## Spotlights
-![](Spotlights\method.png)
-![](Spotlights\literature.png)
-![](Spotlights\data.png)
-![](Spotlights\code.png)
-![](Spotlights\result.png)
## More about author
- ![](Yiwei_photo.jpg)
### Self-introduction
- - Yiwei Liang has extensive experience in computer technology and office software. He is not only familiar with front-end development technologies, such as HTML, CSS, JavaScript, etc., but is also proficient in using programming languages such as Python and Java.
### Final Reflections
- Intellectual Growth: Understand machine learning's societal impact by studying case studies. Analyze ethical considerations. Explore interdisciplinary perspectives to grasp its role in addressing social and economic issues.

- Professional Growth: Acquire technical proficiency in ML and soft skills. Network within diverse communities. Create a balanced professional profile, blending technical expertise with societal understanding.

- Purposeful Life: Aim to pioneer ethical AI for societal good. Envision contributions recognized with a Nobel or Turing Award. Aspire to reshape society positively, prioritizing ethical considerations and advancing human well-being.
## References
### Data Source
- Sentiment Analysis Dataset: https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?rvi=1
### Code Source
- https://github.com/tloen/alpaca-lora
### Literature
- Neethu, M. S., and R. Rajasree. "Sentiment analysis in twitter using machine learning techniques." In 2013 fourth international conference on computing, communications and networking technologies (ICCCNT), pp. 1-5. IEEE, 2013.
- Agarwal, Basant, Namita Mittal, Basant Agarwal, and Namita Mittal. "Machine learning approach for sentiment analysis." Prominent feature extraction for sentiment analysis (2016): 21-45.
- Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. “LoRA: Low-Rank Adaptation of Large Language Models.” ArXiv:2106.09685 [Cs], October. https://arxiv.org/abs/2106.09685.
```
@inproceedings{neethu2013sentiment,
  title={Sentiment analysis in twitter using machine learning techniques},
  author={Neethu, MS and Rajasree, R},
  booktitle={2013 fourth international conference on computing, communications and networking technologies (ICCCNT)},
  pages={1--5},
  year={2013},
  organization={IEEE}
}
@article{agarwal2016machine,
  title={Machine learning approach for sentiment analysis},
  author={Agarwal, Basant and Mittal, Namita and Agarwal, Basant and Mittal, Namita},
  journal={Prominent feature extraction for sentiment analysis},
  pages={21--45},
  year={2016},
  publisher={Springer}
}
@article{hu2021lora,
  title={Lora: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```