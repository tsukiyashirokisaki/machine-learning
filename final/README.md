# ml2019fall final project: Domain Adaptation
## [題目敘述](https://drive.google.com/open?id=1leYCs0RNjYF8sBGc7SuF5-R3awoRF_QF)

## 下載資料的方式:

進到data資料夾裡面，執行down_data.sh這個檔案，則trainX.npy、trainY.npy、testX.npy基本資料會被下載，此外，代表DANN取交集的答案output.npy也會被一併下載，該檔案第一欄為output的答案(0-9)，第二欄代表是否為交集的答案(0-1)。

## 下載模型的方式:

進到data資料夾裡面，執行down_model.sh這個檔案，則DANN的三個模型會model_C_288_gray_86.pkl、model_D_288_gray_86.pkl、model_F_288_gray_86.pkl被載下來，以及半監督式學習部分的模型 Resnet18like.pkl會被載下來。

## DANN_test、Semi_test

先把所有檔案從github下載下來，如果想要使用DANN train好的model(model_C_288_gray_86.pkl、model_F_288_gray_86.pkl)，以及半監督式學習train好的model(Resnet18like.pkl)，執行test.sh，會先做DANN的testing再做半監督式學習的testing，分別生成"DANN_edge_288.csv"以及"semi_ans.csv"在submission資料夾中。

## DANN_train、semi_train
如果想要train model，執行train sh，會先做DANN的training再做半監督式學習的training。DANN存下來的模型會被叫做  model_C_"epoch數"_gray_86.pkl、model_F_"epoch數"_gray_86.pkl、model_D_"epoch數"_gray_86.pkl (結尾一定是86)，位於model這個資料夾中，接著把DANN_test.py第367行(final = predict(288,test_loader))的288改成想要使用的model的epoch數，再執行test.sh，csv檔會生成在submission資料夾中。
Semi存下來的模型會被叫做”model name_validation accuracy(%)_epoch_step.pkl”，model name分為Resnet和CNN，validation accuracy存的是一位浮點數，step指的是該epoch的中第幾批，同時，”model name_validation accuracy(%)_epoch_step.csv”會被生成在submission資料夾中。
