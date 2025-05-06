# My Submission to Kaggle's Titanic Challenge 
[https://www.kaggle.com/competitions/titanic/overview](Challenge's link)

My current model peaked at a score of **0.73205**. All future enhancements would take place in this repository. The model is made up of fully connected neuron layers utilizing cross entropy loss and softmax. In the future, dropouts and feature engineering would be used to improve the score and reliability of the model.

![Loss plot of the model](https://github.com/Arsalanjdev/titanic-kaggle/blob/main/loss.png)
![Accuracy plot of the model](https://github.com/Arsalanjdev/titanic-kaggle/blob/main/acc.png)

## How to Run

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo.git](https://github.com/Arsalanjdev/titanic-kaggle)
cd titanic-kaggle
python main.py
```

The inference of the model on the test.csv file will be saved on survived.csv. The program also generates loss.png and acc.png alongside a tensorboard monitor directory.
