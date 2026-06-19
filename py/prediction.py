import argparse
import os

from ALIDDM_utils import *
from classes import *
import pandas as pd
import GlobVar as GV
from Agent_class import *
from prediction_utils import *
from monai.networks.nets import UNet
import cv2 as cv
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
import shutil
import vtk
from scipy import linalg
import matplotlib.pyplot as plt

def visualize_inputs(inputs, num_cameras, label, patient_info):
    """Visualize the input images for the model"""
    B, C, H, W = inputs.shape
    num_channels = C // num_cameras
    
    fig, axes = plt.subplots(num_cameras, num_channels, figsize=(12, 4*num_cameras))
    if num_cameras == 1:
        axes = axes.reshape(1, -1)
    
    inputs_np = inputs.detach().cpu().numpy()
    
    for cam in range(num_cameras):
        for ch in range(num_channels):
            idx = cam * num_channels + ch
            if idx < C:
                ax = axes[cam, ch]
                img = inputs_np[0, idx, :, :]
                im = ax.imshow(img, cmap='gray')
                ax.set_title(f"Camera {cam}, Channel {ch}")
                ax.axis('off')
                plt.colorbar(im, ax=ax)
    
    fig.suptitle(f"Inputs for {patient_info} - Label {label}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main(args):
    GV.DEVICE = torch.device(f"cuda:{args.num_device}" if torch.cuda.is_available() else "cpu")
    jaw = args.jaw
    
    if jaw == "U":
        lst_label = args.label_U
        dir_model = args.model_U
        csv_file = args.csv_file_U
    else :
        lst_label = args.label_L
        dir_model = args.model_L
        csv_file = args.csv_file_L

    lst_vtkfiles = []


    df = pd.read_csv(csv_file)

    for vtkfile in df['surf']:
        full_vtkfile = os.path.join(args.patient_path, vtkfile)
        lst_vtkfiles.append(full_vtkfile)

    for path_vtk in lst_vtkfiles:
        num_patient = os.path.basename(path_vtk).split('.')[0].split('_')[0]
        scan_period = os.path.basename(path_vtk).split('.')[0].split('_')[1]
        print(f"prediction for patient {num_patient}_{scan_period} :", path_vtk )
        groupe_data = {}
        
        for label in lst_label:
            model = os.path.join(dir_model,f"best_metric_model.pth")
            print("Loading model :", model, "for patient :", num_patient,"_",scan_period, " label :", label)
            phong_renderer,mask_renderer = GenPhongRenderer(args.image_size,args.blur_radius,args.faces_per_pixel,GV.DEVICE)

            agent = Agent(
                renderer=phong_renderer,
                renderer2=mask_renderer,
                radius=args.sphere_radius,
                camera_positions=GV.dic_cam[args.lm_type][jaw]
            )

            SURF = ReadSurf(path_vtk)    
            surf_unit, mean_arr, scale_factor= ScaleSurf(SURF)
            (V, F, CN, RI) = GetSurfProp(surf_unit, mean_arr, scale_factor)
            num_cameras = len(GV.dic_cam[args.lm_type][jaw])
            channels_per_camera = 4  # RGB (3) + Z (1) = 4 channels per camera
            out = 4 if args.lm_type=='O' else 3
            total_in_channels = num_cameras * channels_per_camera
            if int(label) in RI.squeeze(0):
                agent.position_agent(RI,V,label)
                textures = TexturesVertex(verts_features=CN)
                meshe = Meshes(
                            verts=V,   
                            faces=F, 
                            textures=textures
                            ).to(GV.DEVICE)
                images_model , tens_pix_to_face_model=  agent.get_view_rasterize(meshe, args.jaw) #[batch,num_ima,channels,size,size] torch.Size([1, 2, 4, 224, 224])
                tens_pix_to_face_model = tens_pix_to_face_model.permute(1,0,4,2,3) #tens_pix_to_face : torch.Size([1, 2, 1, 224, 224])
                
                net = UNet(
                    spatial_dims=2,
                    in_channels=total_in_channels,
                    out_channels=out,
                    channels=( 16, 32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2, 2),
                    num_res_units=4
                ).to(GV.DEVICE)
                
                # Reshape inputs to match training format: [batch, num_cameras*channels, H, W]
                B, Cam, C, H, W = images_model.shape
                inputs = images_model.reshape(B, Cam * C, H, W).to(dtype=torch.float32).to(GV.DEVICE)
                
                # Visualize the inputs
                patient_info = f"{num_patient}_{scan_period}"
                visualize_inputs(inputs, Cam, label, patient_info)
                
                net.load_state_dict(torch.load(model, weights_only=True))
                images_pred = net(inputs)

                post_pred = AsDiscrete(argmax=True, to_onehot=4)


                val_pred_outputs_list = decollate_batch(images_pred)                
                val_pred_outputs_convert = [
                    post_pred(val_pred_outputs_tensor) for val_pred_outputs_tensor in val_pred_outputs_list
                ]
                val_pred = torch.empty((0)).to(GV.DEVICE)
                for image in images_pred:
                    val_pred = torch.cat((val_pred,post_pred(image).unsqueeze(0).to(GV.DEVICE)),dim=0)
                         
                
                pred_data = images_pred.detach().cpu().unsqueeze(0).type(torch.int16) #torch.Size([1, 2, 2, 224, 224])
                pred_data = torch.argmax(pred_data, dim=2).unsqueeze(2)
      
                # recover where there is the landmark in the image
                index_label_land_r = (pred_data==1.).nonzero(as_tuple=False) #torch.Size([6252, 5])
                index_label_land_g = (pred_data==2.).nonzero(as_tuple=False) #torch.Size([6252, 5])
                index_label_land_b = (pred_data==3.).nonzero(as_tuple=False) #torch.Size([6252, 5])

                print(f"Number of predicted landmark pixels for label {label} - Red: {len(index_label_land_r)}, Green: {len(index_label_land_g)}, Blue: {len(index_label_land_b)}")
                # recover the face in my mesh 
                num_faces_r = []
                num_faces_g = []
                num_faces_b = []
            
                for index in index_label_land_r:
                    num_faces_r.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]]) 
                for index in index_label_land_g:
                    num_faces_g.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]])
                for index in index_label_land_b:
                    num_faces_b.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]]) 
                
                
                last_num_faces_r = remove_extra_faces(F,num_faces_r,RI,int(label))
                last_num_faces_g = remove_extra_faces(F,num_faces_g,RI,int(label))
                last_num_faces_b = remove_extra_faces(F,num_faces_b,RI,int(label))       

                dico_rgb = {}
                dico_rgb[f'{GV.dic_label[args.lm_type][label][0]}'] = last_num_faces_r
                # dico_rgb[f'{GV.dic_label[args.lm_type][label][1]}'] = last_num_faces_g
                if args.lm_type == 'O':
                    dico_rgb[f'{GV.dic_label[args.lm_type][label][2]}'] = last_num_faces_b
                
                
                locator = vtk.vtkOctreePointLocator()
                locator.SetDataSet(surf_unit)
                locator.BuildLocator()
                
                for land_name,list_face_ids in dico_rgb.items():
                    print(land_name)
                    list_face_id=[]
                    for faces in list_face_ids:
                        faces_int = int(faces.item())
                        juan = F[0][faces_int]
                        list_face_id += [int(juan[0].item()) , int(juan[1].item()) , int(juan[2].item())]
                    
                    vert_coord = 0
                    for vert in list_face_id:
                        vert_coord += V[0][vert]
                    if len(list_face_id) != 0:
                        landmark_pos = vert_coord/len(list_face_id)
                        pid = locator.FindClosestPoint(landmark_pos.cpu().numpy())
                        closest_landmark_pos = torch.tensor(surf_unit.GetPoint(pid))

                        upscale_landmark_pos = Upscale(closest_landmark_pos,mean_arr,scale_factor)
                        final_landmark_pos = upscale_landmark_pos
                        
                        coord_dic = {"x":final_landmark_pos[0],"y":final_landmark_pos[1],"z":final_landmark_pos[2]}
                        groupe_data[f'{land_name}']=coord_dic

        lm_lst = GenControlePoint(groupe_data)
        out_path = os.path.join(args.out_path,f"{num_patient}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # out_path_jaw = os.path.join(out_path,os.path.basename(path_vtk).split('.')[0].split('_')[0])
        # if not os.path.exists(out_path_jaw):
        #     os.makedirs(out_path_jaw)
 
        copy_file = os.path.join(out_path,os.path.basename(path_vtk))
        final_out_path = shutil.copy(path_vtk,copy_file)
        
        # landmark_path = os.path.join(os.path.dirname(path_vtk),f"Lower_P{num_patient}.json")
        # if not os.path.exists(landmark_path):
        #     os.makedirs(landmark_path)
        # copy_json_file =  os.path.join(out_path_jaw,os.path.basename(landmark_path))
        # final_outpath_json = shutil.copy(landmark_path,copy_json_file)
        # final_out_path = shutil.copytree(path_vtk,out_path_L)
        jaw_path = "Upper" if args.jaw == "U" else "Lower"
        lm_type_path = args.lm_type 
        WriteJson(lm_lst,os.path.join(out_path,f"{jaw_path}_{num_patient}_{scan_period}_Pred_{lm_type_path}.json"))



def GetSurfProp(surf_unit, surf_mean, surf_scale):     
    surf = ComputeNormals(surf_unit)
    color_normals = ToTensor(dtype=torch.float32, device=GV.DEVICE)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
    verts = ToTensor(dtype=torch.float32, device=GV.DEVICE)(vtk_to_numpy(surf.GetPoints().GetData()))
    faces = ToTensor(dtype=torch.int64, device=GV.DEVICE)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
    region_id = ToTensor(dtype=torch.int64, device=GV.DEVICE)(vtk_to_numpy(surf.GetPointData().GetScalars("Universal_ID")))
    region_id = torch.clamp(region_id, min=0)

    return verts.unsqueeze(0), faces.unsqueeze(0), color_normals.unsqueeze(0), region_id.unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Landmark Identification on Digital Dental Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    # input_param.add_argument('--model_teeth', type=str, help='path of 3D model of the teeth of 1 patient', default='/home/jonas/Desktop/Baptiste_Baquero/data_ALIDDM/data/patients/P20/Lower/Lower_P20.vtk')
    # input_param.add_argument('--vtk_dir', type=str, help='path of 3D model of the teeth of 1 patient', default='/home/luciacev-admin/Desktop/Baptiste_Baquero/Project/ALIDDM/data/Upper_jaw_lab')
    input_param.add_argument('--csv_file_L', type=str, help='path of the csv', default='/home/luciacev/Desktop/training ios files/mucogingival/csv files/data_lower_test_MG.csv')
    input_param.add_argument('--csv_file_U', type=str, help='path of the csv', default='/home/luciacev/Desktop/training ios files/all data/csv files/data_upper_test_O.csv')
    input_param.add_argument('--patient_path', type=str, help='path of the patient folder', default='/home/luciacev/Desktop/training ios files/mucogingival')

    # input_param.add_argument('--model_teeth', type=str, help='path of 3D model of the teeth of 1 patient', default='/Users/luciacev-admin/Desktop/data_ALIDDM/data/Patients /P3/Lower/Lower_P3.vtk')
    # input_param.add_argument('--jsonfile', type=str, help='path of jsonfile of the teeth of 1 patient', default='/home/jonas/Desktop/Baptiste_Baquero/data_ALIDDM/data/patients/P10/Lower/Lower_P10.json')

    # Model directories
    input_param.add_argument('--model_U', type=str, help='loading of model', default='/home/luciacev/Desktop/training ios files/all data/models/Upper/Cervical/fold_0')
    input_param.add_argument('--model_L', type=str, help='loading of model', default='/home/luciacev/Desktop/training ios files/mucogingival/models/Lower/fold_0')

    # Environment
    input_param.add_argument('--jaw',type=str,help="Prepare the data for uper or lower landmark training (ex: L U)", default="L")
    input_param.add_argument('--lm_type',type=str,help="Prepare the data for cervical or occlusal landmark training (ex: O C)", default="MG")
    input_param.add_argument('--sphere_radius', type=float, help='Radius of the sphere with all the cameras', default=0.2)
   
    input_param.add_argument('--label_L', type=list, help='label of the teeth',default=["18","19","20","21","22","23","24","25","26","27","28","29","30","31"])
    input_param.add_argument('--label_U', type=list, help='label of the teeth',default=(["2","3","4","5","6","7","8","9","10","11","12","13","14","15"]))

    # Prediction data
    input_param.add_argument('--num_device',type=str, help='cuda:0 or cuda:1', default='0')
    input_param.add_argument('--image_size',type=int, help='size of the picture', default=224)
    input_param.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    input_param.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
 
    input_param.add_argument('--out_path',type=str, help='path where jsonfile is saved', default='/home/luciacev/Desktop/training ios files/mucogingival/output')

    args = parser.parse_args()
    main(args)