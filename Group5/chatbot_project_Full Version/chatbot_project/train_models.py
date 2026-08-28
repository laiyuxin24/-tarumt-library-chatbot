from engines.svm_engine import train_and_save as train_svm
from engines.st_engine import train_and_save as train_st
from engines.dialogflow_engine import train_and_save as train_dialogflow


if __name__ == "__main__":
    print("\n########## 1) Training SVM + TF-IDF ##########\n")
    train_svm()

    print("\n########## 2) Building Sentence Transformer Embeddings ##########\n")
    train_st()

    print("\n########## 3) Evaluating Dialogflow ##########\n")
    train_dialogflow()

    print("\n✅ All models are ready. You can now run: streamlit run app.py")
