import res_anet
import resnet50

model = resnet50.danet_resnet101(512,512,3,5)
print(model.summary())
