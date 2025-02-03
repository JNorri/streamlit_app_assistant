init:
    pip install -r requirements.txt
    mkdir -p model_data/dataset
    cp kaggle.json model_data/dataset/

download-data:
    cd model_data/dataset && python download_model_data.py

preprocess:
    cd model_data/model && python load_and_preprocess.py

train:
    cd model_data/model && python train_model.py

run-app:
    cd streamlit_app && streamlit run app/app.py

full-setup: init download-data preprocess train

clean:
    rm -rf model_data/dataset/train
    rm -rf model_data/dataset/test
    rm -rf model_data/dataset/valid
    rm -rf model_data/model/train_model/*