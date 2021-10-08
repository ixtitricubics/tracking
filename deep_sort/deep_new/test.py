import os
import cv2
import torchvision.transforms as transforms
from model import *
from skimage import io
from skimage import color
img_path = '/home/ixti/Documents/projects/datasets/action_det_mine/clips/pickup/002/'

model_load_path = 'model_best_35classes.pth'
num_classes = 35
thres = 0.8
ImgSize = (256,256)

model = MobileNet_Large(train_fe=False,classes=num_classes)
checkpoint = torch.load(model_load_path)
model.load_state_dict(checkpoint['state_dict'])

    
linear_output = None

def linear_hook(module, input_, output):
    global linear_output
    linear_output = output

model.linear[0].register_forward_hook(linear_hook)



model = model.cuda()
model = nn.DataParallel(model)


trans = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


model.eval()
with torch.no_grad():
    lst_file = os.listdir(img_path)
    lst_file = sorted(lst_file)
    fn = transforms.ToTensor()
    for img_name in lst_file:
        im = io.imread(os.path.join(img_path, img_name))
        im = cv2.resize(im, ImgSize)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, c = im.shape
        if c==4:
            im = color.rgba2rgb(im)
        elif c==1:
            im = color.gray2rgb(im)
        print("before0", np.array(im).max(), np.array(im).min())
        img_tensor = fn(np.uint8(im))
        print("before", np.array(img_tensor).max(), np.array(img_tensor).min())
        img_tensor = trans(img_tensor)
        print("after",np.array(img_tensor).max(), np.array(img_tensor).min())
        model(img_tensor.unsqueeze(0).cuda())
        # print(linear_output.shape)

        mins = torch.min(img_tensor,dim=0, keepdim=True)
        # min_val = torch.min(img_tensor)
        print(mins)