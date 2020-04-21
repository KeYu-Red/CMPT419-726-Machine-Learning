• The command to run my code:

conda env create -f cmpt726-pytorch-python36.yml
source activate cmpt726-pytorch-python36
python3.7 cifar_finetune.py

• Modification and explanation:
1. In order to speed up the training, I change the code to train on GPU instead of CPU. I used 'cuda' to accomplish it.
Changes:
line 134:  model = model.cuda()
line 143:  criterion = criterion.cuda()
line 151:  nputs = inputs.cuda()
line 152:  abels = labels.cude()

2. I chose the third task: "Try applying L2 regularization to the coefficients in the small networks we added."
From the torch document:
  torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0,weight_decay=0, nesterov=False)
The parameter: weight_decay (python:float, optional) – weight decay (L2 penalty) (default: 0)

The modification is in line 144. I add a parameter value weight_decay = 0.5 :
line 144:  optimizer = optim.SGD(list(model.fc.parameters()), lr=0.001, momentum=0.9, weight_decay=0.5)
The weight_decay arugment here decided the value of L2 regularizer.


