# Real-Time-Facial-Expression-Detection-Using-Machine-Learning
Facial Expression Recognition (FER) provides an invaluable technology useful for enhancing the interactions between humans and machines, and for performing emotion analytics. The aim of this project is developing a system that classifies human facial expressions into Happy, Sad, and Neutral states, with the help of DISFA dataset. Data preprocessing, feature engineering and model training based on Random Forest Classifier was the workflow that was followed. Because of the dataset balancing and adding some more live images, the system reached an accuracy of above 90% and proved to work in real-time on the edge devices.

Keywords : Facial Expression Recognition, DISFA , Machine Learning, Random Forest Classifier
Problem Statement

Our project is creating a smart system that captures human emotions through facial expressions, which is more difficult compared to how it sounds. It is often the case that existing technologies do not perform very well in understanding what people tend to feel when looking at faces.

Let us focus on the three prominent issues. First, in most known systems they have a problem of having unbalanced training data in that they may train on one emotion and be able to identify that one but not others. Second, it should be a fast system with minimal time lag unlike the existing systems. Most importantly, we want to be able to see those micro-expressions that give away the emotion someone is trying to suppress. Not every important feeling is expressed in a strong and obvious way. We refer to those feelings as strong and target to achieve a system that can be capable of identifying such minute subcutaneous movements of muscles.

We are first of all addressing challenges and by doing so in an effective manner, we are going a step further on the journey as well. It is technology that has the ability to decipher emotions and finally bring machines to a point where they can actually see and feel.
Objectives

    Dataset Development: Developing a comprehensive and balanced emotional dataset involves the strategic combination of two key sources:
    DISFA Dataset: This is a foundational set of facial expression data used for scientifically validated baseline generation.
    Custom Live Image Collection: Original images taken for diversity augmentation of the dataset by filling gaps of DISFA and real-world variability of facial expressions.

2. The hybrid approach used in this system ensures all happy, sad, and neutral states of emotions. There may be potential limitations within an existing database.

3. Model Design and Training: We are looking to design a sophisticated machine learning model engineered for:
High accuracy in emotion recognition
Optimized real-time performance
Efficient computational processing

4. Prototype: Deploy the trained model on an edge device-the Jetson Nano-to demonstrate in practice the actual applicability of the system. This allows the model to execute sophisticated emotion recognition on machines with limited resources and plug the gap between advanced research and practical technological solutions.
Expected Outcomes

Our research aims to deliver transformative results in emotion recognition technology

    Accuracy in Emotion Detection: We will design a system that has classification accuracy greater than 90%. This high-performance threshold means our model will reliably and consistently identify emotional states, transforming how machines understand human facial expressions.
    Technological Adaptability: We shall be able to show just how efficient the system really is by implementing our solution successfully on resource-constrained edge devices such as the Jetson Nano. This capability opens up new possibilities in portable and embedded intelligent systems: even advanced emotion recognition could be possible using limited resources.
    Scientific and Technological Advancement: Beyond technical performance, our project will contribute critical insights into the complex field of emotion recognition. We’ll provide a comprehensive framework for:
    ◦ Addressing dataset imbalance challenges
    ◦ Developing techniques for detecting and interpreting subtle emotional signals
    ◦ More flexible and sensitive machine learning approaches to understanding human emotion

These outcomes represent more than incremental progress — they signify a meaningful step towards bridging the gap between computational analysis and the rich, complex world of human emotional expression.
Data Collection and Preprocessing

Our dataset is the result of a strategic balance between avail-abilities of existing research resources and original data collection. We started from the base reference of the DISFA dataset and complemented it with targeted live recordings.
We recorded live video sessions with three participants, each expressing Happy, Sad, and Neutral emotional states over one-minute video sequences. Using Open CV, we have carefully extracted individual frames from these recordings to create a raw visual repository of human emotional expressions. The DeepFace framework has been particularly useful for our initial labeling of the frames. We executed an automated emotion detection algorithm that labeled each frame in its respective category. However, recognizing the potential for computational bias, we implemented a rigorous manual verification process. Our team carefully reviewed and validated each frame, ensuring precise emotional categorization.
A strict filtering mechanism was enforced for maintaining the integrity of our dataset. All the frames that were not correctly depicted Happy, Sad, or Neutral were systematically removed, which ensured that the curations made were always about high-quality and reliable trainings.
We used data augmentation techniques to enhance dataset diversity and model robustness. With horizontal flipping and slight rotational variations, we expanded our original collection to a balanced dataset of 8,000 images per emotional category.
This has resulted in a very well-curated, diverse dataset of the subtle spectrum of human emotional expressions, especially for training an advanced facial emotion recognition model.
Model Selection and Justification

This model was selected based on the fact that the Random Forest Classifier has a good balance between strength, interpret ability, and computation requirements. Therefore, it can work better in resource-constrained devices, which is suitable for deploying at the edge of things. It also provides solutions for the challenges that exist within an imbalanced dataset; because the ensemble averages result from multiple decision trees, thus minimizing bias and giving high accuracy overall, the proposed system is very reliable.
One of the benefits of Random Forest is its handling of complex, non-linear relationships among features, which is essential in picking up subtle cues of emotions from facial expressions. The robustness within the model is a means of consistent performance on multiple datasets, even if some individual features are insignificant at various times.
Moreover, Random Forest allows for clear insights into the feature importance, giving more insight into which facial features contribute most to the classification of Happy, Sad, and Neutral expressions. Transparency is particularly valuable in validating model decisions and refining the dataset.
Compared to Alternatives like Decision Trees, which likely tend to over fit to specific patterns in the training data, Random Forest somewhat negates this problem in itself through ensemble learning. Meanwhile, SVMs usually require careful tuning and are unwieldy for large multiclass datasets. Random Forest provides higher accuracy, reduced odds of overfitting compared with the alternatives, as well as practical ease for any implementation.
With the strengths combined, the Random Forest Classifier will provide reliable detection in facial expressions in real time-a great choice for this project.

Model Training
Random Forest Classifier
100 decision trees for classification
Train on combined HOG + LBP features
Fit the model using training data (X_train_combined, y_train)
Model Training, Evaluation, and Metrics

The Random Forest Classifier was trained on a balanced dataset of the three emotion categories, namely Happy, Sad, and Neutral. It had to be tuned through the training process for parameters like the number of estimators and tree depth for maximizing the performance of the model using a grid search approach. This way, it could well be tuned for its sensitivity towards the nuances of facial expressions and computational efficiency at the same time.
For comprehensive assessment of the model, several metrics were used. Accuracy referred to the general classification ability, while Precision and Recall gave insights into how good the model is for every emotion category. F1 Score was used for balancing precision and recall, mainly to deal with class imbalances. Also, a confusion matrix was used to spot areas where the model might have been mistaken and, in this case, the “Sad”, “neutral” category was always a point of concern as it is quite tricky and people tend to make such expressions.

The model demonstrated strong results across the metrics. The Precision, Recall, and F1 Score for each category were as follows

These results highlight the model’s robustness in recognizing Happy and Neutral expressions, while also showcasing reasonable performance for the more challenging Sad category. The systematic evaluation confirmed the model’s capability to achieve high accuracy and reliable classification, making it suitable for practical deployment on edge devices.
Interpretation, Insights, Presentation, and Documentation
Model Insights

The model was quite strong, including high precision in identifying Happy and Neutral expressions and performing efficiently in real time on the Jetson Nano. It struggled with recognizing rather subtle Sad expressions, however, and needed more data augmentation and delicate manual verification to improve performance.
Implications

The research reveals a promising direction for emotion recognition technology into critical domains:
Human-Computer Interaction: Innovative and adaptive systems that can dynamically interpret and respond to emotional states, fundamentally changing the design of user experience.
Mental health support: Developing screening tools able to identify early signs of emotional distress, potentially revolutionizing proactive mental health monitoring and intervention strategies.
Security and Situational Awareness: Developed intelligent surveillance mechanisms which can detect and analyze unusual emotional patterns to enhance safety protocols and threat detection capabilities.
Limitations

While our model shows promising performance, it faces serious limitations and these constitute the direction for future work:
1. Emotional Domain Extension: This dataset so far only provides three dimensions of emotions — Happy, Sad, Neutral. Successive versions will have more comprehensive emotional variations to fit into the application of models in real-world scenarios.
2. Environmental Robustness: The model’s performance under various environmental conditions may vary. Research in the next stage should emphasize algorithms that are robust across lighting, background, and contextual variations.
Conclusion and Future Work

This project culminated into the development of a very accurate and real-time facial expression recognition system. Using the DISFA dataset and supplementing it with live images helped the team deal with challenges related to the quality and diversity of datasets. Deployment of the model on an edge device such as the Jetson Nano showed practical applicability and efficiency, reinforcing its potential for use in real-world applications.
The project demonstrated strong performance in recognizing Happy, Sad, and Neutral expressions while portraying critical insights in handling imbalanced data, subtle emotion detection, and resource constraints. All these accomplishments set a strong foundation for future enhancement.
The system can be extended to recognize a wider range of emotions, thus making it more versatile for different applications. Further testing under different real-world conditions, such as changing lighting and diverse backgrounds, would make it more reliable. Integration into larger systems, such as adaptive interfaces or mental health monitoring tools, could open up new opportunities for impactful applications.
This work represents a key step in advancing emotion recognition technology towards bridging the gap between computational efficiency and practical deployment in everyday scenarios.
Future Directions

The path forward for our emotion recognition model is one of strategic enhancement and commitment to full development. Our primary focus will be on dataset expansion, in an effort to capture a more nuanced and inclusive representation of human emotional landscapes. By intentionally broadening our emotional spectrum, we seek to create a system that can discern and interpret a rich tapestry of human experiences with unprecedented depth and sensitivity.
As the complexity of our model architecture evolves strategically in the direction of Convolutional Neural Networks, there is remarkable feature extraction along with computational scalability. A shift from this will provide not just incremental improvements but a fundamental re imagining of our approach to artificial systems and emotional intelligence. As our dataset grows with increasing complexity and diversity, Cans will provide that robust computational framework necessary for transforming raw visual data into meaningful emotional insights.
Complementing these technological advancements, our development process will emphasize rigorous, real-world validation. Comprehensive testing protocols will challenge the model across a spectrum of environmental conditions, from challenging lighting scenarios to complex visual landscapes with occlusions and dynamic backgrounds. This approach will ensure that our system moves beyond the confines of a laboratory to emerge as a resilient, adaptable solution with consistent performance in the uncertain terrain of real-world applications.
By expanding our dataset, refining our neural architecture, and stress-testing our model’s capabilities at the same time, we are not just developing a technical solution but rather constructing a sophisticated instrument of emotional understanding that bridges the gap between computational precision and human complexity.
References

    Mavadati, S. M., Mahoor, M. H., Bartlett, K., Trinh, P., & Cohn, J. F. (2013). DISFA: A spontaneous facial action intensity database. IEEE Transactions on Affective Computing, 4(2), 151–160. DOI: 10.1109/T-AFFC.2013.4
    Huang, ZY., Chiang, CC., Chen, JH. et al. A study on computer vision for facial emotion recognition. Sci Rep 13, 8425 (2023). https://doi.org/10.1038/s41598-023-35446-4
    B. Fang, X. Li, G. Han and J. He, “Facial Expression Recognition in Educational Research From the Perspective of Machine Learning: A Systematic Review,” in IEEE Access, vol. 11, pp. 112060–112074, 2023, doi: 10.1109/ACCESS.2023.3322454.:https://ieeexplore.ieee.org/document/10273682
