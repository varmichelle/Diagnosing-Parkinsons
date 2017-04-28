# A Machine Learning Approach for the Diagnosis of Parkinson's Disease via Speech Analysis

## Introduction
- Parkinson’s Disease is the second most prevalent neurodegenerative disorder after Alzheimer’s, affecting more than 10 million people worldwide. Parkinson’s is characterized primarily by the deterioration of motor and cognitive ability.
- There is no single test which can be administered for diagnosis. Instead, doctors must perform a careful clinical analysis of the patient’s medical history. 
- Unfortunately, this method of diagnosis is highly inaccurate. A study from the National Institute of Neurological Disorders finds that early diagnosis (having symptoms for 5 years or less) is only 53% accurate. This is not much better than random guessing, but an early diagnosis is critical to effective treatment.
- Because of these difficulties, I investigate a machine learning approach to accurately diagnose Parkinson’s, using a dataset of various speech features (a non-invasive yet characteristic tool) from the University of Oxford.
- WHY SPEECH FEATURES? Speech is very predictive and characteristic of Parkinson’s disease; almost every Parkinson’s patient experiences severe vocal degradation (inability to produce sustained phonations, tremor, hoarseness), so it makes sense to use voice to diagnose the disease. Voice analysis gives the added benefit of being non-invasive, inexpensive, and very easy to extract clinically.

## Background
### Parkinson's Disease
- Parkinson’s is a progressive neurodegenerative condition resulting from the death of the dopamine containing cells of the substantia nigra (which plays an important role in movement). 
- Symptoms include: “frozen” facial features, bradykinesia (slowness of movement), akinesia (impairment of voluntary movement), tremor, and voice impairment.
- Typically, by the time the disease is diagnosed, 60% of nigrostriatal neurons have degenerated, and 80% of striatal dopamine have been depleted. 

### Performance Metrics
- * TP = true positive, FP = false positive, TN = true negative, FN = false negative
- Accuracy: (TP+TN)/(P+N)
- Matthews Correlation Coefficient: 1=perfect, 0=random, -1=completely inaccurate 

