import json
from posixpath import basename
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import numpy as np
import glob
import csv

def Upscale(landmark_pos,mean_arr,scale_factor):
    new_pos_center = (landmark_pos.cpu()/scale_factor) + mean_arr
    return new_pos_center

def GenControlePoint(groupe_data):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark,data in groupe_data.items():
        id+=1
        controle_point = {
            "id": str(id),
            "label": landmark,
            "description": "",
            "associatedNodeID": "",
            "position": [float(data["x"]), float(data["y"]), float(data["z"])],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": true,
            "locked": true,
            "visibility": true,
            "positionStatus": "preview"
        }
        lm_lst.append(controle_point)

    return lm_lst

def WriteJson(lm_lst,out_path):
    false = False
    true = True
    file = {
    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
    "markups": [
        {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "locked": false,
            "labelFormat": "%N-%d",
            "controlPoints": lm_lst,
            "measurements": [],
            "display": {
                "visibility": false,
                "opacity": 1.0,
                "color": [0.4, 1.0, 0.0],
                "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": false,
                "pointLabelsVisibility": true,
                "textScale": 3.0,
                "glyphType": "Sphere3D",
                "glyphScale": 1.0,
                "glyphSize": 5.0,
                "useGlyphScale": true,
                "sliceProjection": false,
                "sliceProjectionUseFiducialColor": true,
                "sliceProjectionOutlinedBehindSlicePlane": false,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": false,
                "snapMode": "toVisibleSurface"
            }
        }
    ]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        # print(file)
        json.dump(file, f, ensure_ascii=False, indent=4)

    f.close

def ReadJson(fiducial_path):
    lm_dic = {}
    with open(fiducial_path) as f:
            data = json.load(f)
    markups = data["markups"][0]["controlPoints"]
    for markup in markups:
        lm_dic[markup["label"]] = {"x":markup["position"][0],"y":markup["position"][1],"z":markup["position"][2]}
    return lm_dic

def classify_landmark(lm):
    """Classify landmark as 'Occlusal' or 'Cervical'"""
    if any(lm.endswith(suffix) for suffix in ['O', 'MB', 'DB']):
        return 'Occlusal'
    elif any(lm.endswith(suffix) for suffix in ['CB', 'CL']):
        return 'Cervical'
    else:
        return 'Unknown'

def get_specific_type(lm):
    """Get the specific type of landmark (O, MB, DB, CB, CL)"""
    if lm.endswith('O'):
        return 'O'
    elif lm.endswith('MB'):
        return 'MB'
    elif lm.endswith('DB'):
        return 'DB'
    elif lm.endswith('CB'):
        return 'CB'
    elif lm.endswith('CL'):
        return 'CL'
    else:
        return 'Unknown'

def ResultAccuracy(fiducial_dir):

    error_dic = {"labels":[], "error":[]}
    patients = {}
    normpath = os.path.normpath("/".join([fiducial_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(img_fn) and ".json" in img_fn:
            baseName = os.path.basename(img_fn)
            patient = os.path.dirname(img_fn)

            if baseName[0] == "L" or baseName[0] == "U":
                num_label_pred = os.path.basename(img_fn).split('_')[1]+"_"+os.path.basename(img_fn).split('_')[2]+"_"+os.path.basename(img_fn).split('_')[-1]
            else:
                num_label_pred = os.path.basename(img_fn).split('_')[0]+"_"+os.path.basename(img_fn).split('_')[1]+"_"+os.path.basename(img_fn).split('_')[-1]

            if patient not in patients.keys():
                patients[patient] = {"Upper":{},"Lower":{}}
            if "_Pred" in baseName:
                if "Upper_" in baseName :
                    patients[patient]["Upper"][f"pred_{num_label_pred}"]=img_fn
                elif "Lower_" in baseName :
                    patients[patient]["Lower"][f"pred_{num_label_pred}"]=img_fn
               
            else:
                if "Upper_" in baseName :
                    patients[patient]["Upper"][f"target_{num_label_pred}"]=img_fn

                elif "Lower_" in baseName :
                    patients[patient]["Lower"][f"target_{num_label_pred}"]=img_fn
                
    # Statistics dictionaries
    stats = {
        'Occlusal': {'errors': [], 'fail': 0, 'success': 0},
        'Cervical': {'errors': [], 'fail': 0, 'success': 0},
        'Unknown': {'errors': [], 'fail': 0, 'success': 0}
    }
    specific_stats = {
        'O': {'errors': [], 'fail': 0, 'success': 0},
        'MB': {'errors': [], 'fail': 0, 'success': 0},
        'DB': {'errors': [], 'fail': 0, 'success': 0},
        'CB': {'errors': [], 'fail': 0, 'success': 0},
        'CL': {'errors': [], 'fail': 0, 'success': 0},
        'Unknown': {'errors': [], 'fail': 0, 'success': 0}
    }
    # Statistics by Upper/Lower
    upper_lower_stats = {
        'Upper': {
            'Occlusal': {'errors': [], 'fail': 0, 'success': 0},
            'Cervical': {'errors': [], 'fail': 0, 'success': 0},
            'Unknown': {'errors': [], 'fail': 0, 'success': 0}
        },
        'Lower': {
            'Occlusal': {'errors': [], 'fail': 0, 'success': 0},
            'Cervical': {'errors': [], 'fail': 0, 'success': 0},
            'Unknown': {'errors': [], 'fail': 0, 'success': 0}
        }
    }
    # Specific type stats by Upper/Lower
    upper_lower_specific_stats = {
        'Upper': {
            'O': {'errors': [], 'fail': 0, 'success': 0},
            'MB': {'errors': [], 'fail': 0, 'success': 0},
            'DB': {'errors': [], 'fail': 0, 'success': 0},
            'CB': {'errors': [], 'fail': 0, 'success': 0},
            'CL': {'errors': [], 'fail': 0, 'success': 0},
            'Unknown': {'errors': [], 'fail': 0, 'success': 0}
        },
        'Lower': {
            'O': {'errors': [], 'fail': 0, 'success': 0},
            'MB': {'errors': [], 'fail': 0, 'success': 0},
            'DB': {'errors': [], 'fail': 0, 'success': 0},
            'CB': {'errors': [], 'fail': 0, 'success': 0},
            'CL': {'errors': [], 'fail': 0, 'success': 0},
            'Unknown': {'errors': [], 'fail': 0, 'success': 0}
        }
    }
    patient_stats = {}
    
    # Dictionary to track landmarks by Upper/Lower
    landmark_upper_lower_stats = {
        'Upper': {},
        'Lower': {}
    }
    
    print(patients)
    
    for patient,fiducials in patients.items():
        patient_name = os.path.basename(patient)
        patient_stats[patient_name] = {
            'Upper': {'Occlusal': {'errors': [], 'fail': 0}, 'Cervical': {'errors': [], 'fail': 0}},
            'Lower': {'Occlusal': {'errors': [], 'fail': 0}, 'Cervical': {'errors': [], 'fail': 0}}
        }
        
        print(f"\n{'='*60}")
        print(f"Results for patient: {patient_name}")
        print(f"{'='*60}")
        
        for group,targ_res in fiducials.items():
            print(f"\n  {group} landmarks:")
            
            # Extract unique predictions by matching target and pred pairs
            pred_keys = [k for k in targ_res.keys() if k.startswith("pred_")]
            for pred_key in pred_keys:
                # Replace "pred_" with "target_" to find the corresponding target
                target_key = pred_key.replace("pred_", "target_")
                if target_key in targ_res.keys():
                    target_lm_dic = ReadJson(targ_res[target_key])
                    pred_lm_dic = ReadJson(targ_res[pred_key])
                    
                    # Separate by landmark type
                    landmarks_by_type = {'Occlusal': [], 'Cervical': []}
                    for lm,t_data in target_lm_dic.items():
                        if lm in pred_lm_dic.keys():
                            lm_type = classify_landmark(lm)
                            specific_type = get_specific_type(lm)
                            a = np.array([float(t_data["x"]),float(t_data["y"]),float(t_data["z"])])
                            p_data = pred_lm_dic[lm]
                            b = np.array([float(p_data["x"]),float(p_data["y"]),float(p_data["z"])])
                            dist = np.linalg.norm(a-b)
                            
                            landmarks_by_type[lm_type].append({'name': lm, 'error': dist})
                            error_dic["labels"].append(lm)
                            error_dic["error"].append(dist)
                            
                            # Track by Upper/Lower
                            if lm not in landmark_upper_lower_stats[group]:
                                landmark_upper_lower_stats[group][lm] = {'errors': []}
                            landmark_upper_lower_stats[group][lm]['errors'].append(dist)
                            
                            # Update stats
                            if dist < 5:
                                stats[lm_type]['errors'].append(dist)
                                stats[lm_type]['success'] += 1
                                specific_stats[specific_type]['errors'].append(dist)
                                specific_stats[specific_type]['success'] += 1
                                patient_stats[patient_name][group][lm_type]['errors'].append(dist)
                                # Update Upper/Lower stats
                                upper_lower_stats[group][lm_type]['errors'].append(dist)
                                upper_lower_stats[group][lm_type]['success'] += 1
                                upper_lower_specific_stats[group][specific_type]['errors'].append(dist)
                                upper_lower_specific_stats[group][specific_type]['success'] += 1
                            else:
                                stats[lm_type]['fail'] += 1
                                specific_stats[specific_type]['fail'] += 1
                                patient_stats[patient_name][group][lm_type]['fail'] += 1
                                # Update Upper/Lower stats
                                upper_lower_stats[group][lm_type]['fail'] += 1
                                upper_lower_specific_stats[group][specific_type]['fail'] += 1
                    
                    # Print results by type
                    for lm_type in ['Occlusal', 'Cervical']:
                        if landmarks_by_type[lm_type]:
                            type_name = "Occlusal (O/MB/DB)" if lm_type == 'Occlusal' else "Cervical (CB/CL)"
                            print(f"    {pred_key.split('_', 1)[1]} - {type_name}:")
                            for item in landmarks_by_type[lm_type]:
                                status = "✓" if item['error'] < 5 else "✗"
                                print(f"      {status} {item['name']}: {item['error']:.4f}")
    
    # Print summary statistics by general type (Occlusal vs Cervical)
    print(f"\n\n{'='*60}")
    print("SUMMARY STATISTICS - By Type (Occlusal vs Cervical)")
    print(f"{'='*60}")
    
    for lm_type in ['Occlusal', 'Cervical']:
        print(f"\n{lm_type} Landmarks:")
        errors = stats[lm_type]['errors']
        if errors:
            print(f"  Success: {stats[lm_type]['success']}")
            print(f"  Fail (>5mm): {stats[lm_type]['fail']}")
            print(f"  Mean Error: {np.mean(errors):.4f} mm")
            print(f"  Std Dev: {np.std(errors):.4f} mm")
            print(f"  Min Error: {np.min(errors):.4f} mm")
            print(f"  Max Error: {np.max(errors):.4f} mm")
    
    # Per-specific type summary (O, MB, DB, CB, CL)
    print(f"\n\n{'='*60}")
    print("SUMMARY STATISTICS - By Specific Type (O, MB, DB, CB, CL)")
    print(f"{'='*60}")
    
    for lm_type in ['O', 'MB', 'DB', 'CB', 'CL']:
        print(f"\n{lm_type} Landmarks:")
        errors = specific_stats[lm_type]['errors']
        if errors:
            print(f"  Success: {specific_stats[lm_type]['success']}")
            print(f"  Fail (>5mm): {specific_stats[lm_type]['fail']}")
            print(f"  Mean Error: {np.mean(errors):.4f} mm")
            print(f"  Std Dev: {np.std(errors):.4f} mm")
            print(f"  Min Error: {np.min(errors):.4f} mm")
            print(f"  Max Error: {np.max(errors):.4f} mm")
        else:
            print(f"  No data available")
    
    # Statistics by Upper/Lower and type
    print(f"\n\n{'='*60}")
    print("SUMMARY STATISTICS - By Upper/Lower and Type")
    print(f"{'='*60}")
    
    for jaw in ['Upper', 'Lower']:
        print(f"\n{jaw} Jaw - General Types:")
        for lm_type in ['Occlusal', 'Cervical']:
            errors = upper_lower_stats[jaw][lm_type]['errors']
            if errors:
                print(f"  {lm_type}:")
                print(f"    Success: {upper_lower_stats[jaw][lm_type]['success']}")
                print(f"    Fail (>5mm): {upper_lower_stats[jaw][lm_type]['fail']}")
                print(f"    Mean Error: {np.mean(errors):.4f} mm")
                print(f"    Std Dev: {np.std(errors):.4f} mm")
    
    # Specific types by Upper/Lower
    print(f"\n\n{'='*60}")
    print("SUMMARY STATISTICS - Upper/Lower by Specific Type (O, MB, DB, CB, CL)")
    print(f"{'='*60}")
    
    for jaw in ['Upper', 'Lower']:
        print(f"\n{jaw} Jaw - Specific Types:")
        for lm_type in ['O', 'MB', 'DB', 'CB', 'CL']:
            errors = upper_lower_specific_stats[jaw][lm_type]['errors']
            if errors:
                print(f"  {lm_type}: {len(errors)} landmarks, Mean: {np.mean(errors):.4f}mm")
    
    
    
    # Save summary statistics to CSV
    csv_path = os.path.join(fiducial_dir, "accuracy_summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Landmark_Type', 'Success', 'Fail', 'Mean_Error_mm', 'Std_Dev_mm', 'Min_Error_mm', 'Max_Error_mm', 'Total'])
        
        # Write general type stats (Occlusal vs Cervical)
        for lm_type in ['Occlusal', 'Cervical']:
            errors = stats[lm_type]['errors']
            if errors:
                writer.writerow([
                    lm_type,
                    stats[lm_type]['success'],
                    stats[lm_type]['fail'],
                    f"{np.mean(errors):.4f}",
                    f"{np.std(errors):.4f}",
                    f"{np.min(errors):.4f}",
                    f"{np.max(errors):.4f}",
                    len(errors) + stats[lm_type]['fail']
                ])
        
        # Blank row
        writer.writerow([])
        
        # Write specific type stats (O, MB, DB, CB, CL)
        writer.writerow(['Landmark_Type', 'Success', 'Fail', 'Mean_Error_mm', 'Std_Dev_mm', 'Min_Error_mm', 'Max_Error_mm', 'Total'])
        for lm_type in ['O', 'MB', 'DB', 'CB', 'CL']:
            errors = specific_stats[lm_type]['errors']
            if errors:
                writer.writerow([
                    lm_type,
                    specific_stats[lm_type]['success'],
                    specific_stats[lm_type]['fail'],
                    f"{np.mean(errors):.4f}",
                    f"{np.std(errors):.4f}",
                    f"{np.min(errors):.4f}",
                    f"{np.max(errors):.4f}",
                    len(errors) + specific_stats[lm_type]['fail']
                ])
        
        # Blank rows for separation
        writer.writerow([])
        writer.writerow([])
        
        # Write per-landmark stats
        writer.writerow(['LANDMARK STATISTICS BY LABEL'])
        writer.writerow(['Landmark_Label', 'Success', 'Fail', 'Mean_Error_mm', 'Std_Dev_mm', 'Min_Error_mm', 'Max_Error_mm', 'Total'])
        
        # Calculate stats per landmark
        landmark_stats = {}
        for i, label in enumerate(error_dic['labels']):
            error_val = error_dic['error'][i]
            if label not in landmark_stats:
                landmark_stats[label] = {'errors': []}
            landmark_stats[label]['errors'].append(error_val)
        
        # Write stats for each landmark (sorted by label)
        for label in sorted(landmark_stats.keys()):
            errors = landmark_stats[label]['errors']
            success = np.sum(np.array(errors) < 5)
            fail = np.sum(np.array(errors) >= 5)
            writer.writerow([
                label,
                success,
                fail,
                f"{np.mean(errors):.4f}",
                f"{np.std(errors):.4f}",
                f"{np.min(errors):.4f}",
                f"{np.max(errors):.4f}",
                len(errors)
            ])
        
        # Blank rows for separation
        writer.writerow([])
        writer.writerow([])
        
        # Write Upper/Lower stats by general type
        writer.writerow(['UPPER/LOWER STATISTICS - By General Type'])
        writer.writerow(['Jaw', 'Landmark_Type', 'Success', 'Fail', 'Mean_Error_mm', 'Std_Dev_mm', 'Min_Error_mm', 'Max_Error_mm', 'Total'])
        for jaw in ['Upper', 'Lower']:
            for lm_type in ['Occlusal', 'Cervical']:
                errors = upper_lower_stats[jaw][lm_type]['errors']
                if errors:
                    writer.writerow([
                        jaw,
                        lm_type,
                        upper_lower_stats[jaw][lm_type]['success'],
                        upper_lower_stats[jaw][lm_type]['fail'],
                        f"{np.mean(errors):.4f}",
                        f"{np.std(errors):.4f}",
                        f"{np.min(errors):.4f}",
                        f"{np.max(errors):.4f}",
                        len(errors) + upper_lower_stats[jaw][lm_type]['fail']
                    ])
        
        # Blank rows for separation
        writer.writerow([])
        writer.writerow([])
        
        # Write Upper/Lower stats by specific type
        writer.writerow(['UPPER/LOWER STATISTICS - By Specific Type (O, MB, DB, CB, CL)'])
        writer.writerow(['Jaw', 'Landmark_Type', 'Success', 'Fail', 'Mean_Error_mm', 'Std_Dev_mm', 'Min_Error_mm', 'Max_Error_mm', 'Total'])
        for jaw in ['Upper', 'Lower']:
            for lm_type in ['O', 'MB', 'DB', 'CB', 'CL']:
                errors = upper_lower_specific_stats[jaw][lm_type]['errors']
                if errors:
                    writer.writerow([
                        jaw,
                        lm_type,
                        upper_lower_specific_stats[jaw][lm_type]['success'],
                        upper_lower_specific_stats[jaw][lm_type]['fail'],
                        f"{np.mean(errors):.4f}",
                        f"{np.std(errors):.4f}",
                        f"{np.min(errors):.4f}",
                        f"{np.max(errors):.4f}",
                        len(errors) + upper_lower_specific_stats[jaw][lm_type]['fail']
                    ])
    
    print(f"\n\nResults saved to: {csv_path}")
    
    return error_dic


def PlotResults(data):
    sns.set_theme(style="whitegrid")
    
    # Convert to pandas DataFrame if it's a dict with lists
    import pandas as pd
    if isinstance(data, dict) and 'labels' in data and 'error' in data:
        df = pd.DataFrame(data)
    else:
        df = data
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Original plot: Violin plot by labels
    ax1 = plt.subplot(2, 3, 1)
    ax1 = sns.violinplot(x="labels", y="error", data=df, cut=0, ax=ax1)
    ax1.set_title("Error Distribution by Landmark Label", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Landmark Label", fontsize=10)
    ax1.set_ylabel("Error (mm)", fontsize=10)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Box plot by labels
    ax2 = plt.subplot(2, 3, 2)
    ax2 = sns.boxplot(x="labels", y="error", data=df, ax=ax2)
    ax2.set_title("Error Range by Landmark Label", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Landmark Label", fontsize=10)
    ax2.set_ylabel("Error (mm)", fontsize=10)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Histogram of all errors
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(df['error'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.set_title("Error Distribution (All Landmarks)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Error (mm)", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.axvline(np.mean(df['error']), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(df['error']):.3f}mm")
    ax3.axvline(np.median(df['error']), color='green', linestyle='--', linewidth=2, label=f"Median: {np.median(df['error']):.3f}mm")
    ax3.legend()
    
    # 4. Classify landmarks by type for summary statistics
    landmarks_by_type = {'O': [], 'MB': [], 'DB': [], 'CB': [], 'CL': []}
    for i, lm in enumerate(df['labels']):
        lm_type = get_specific_type(lm)
        if lm_type in landmarks_by_type:
            landmarks_by_type[lm_type].append(df['error'].iloc[i])
    
    # 5. Bar plot of mean errors by type
    ax4 = plt.subplot(2, 3, 4)
    types = []
    means = []
    stds = []
    for lm_type in ['O', 'MB', 'DB', 'CB', 'CL']:
        if landmarks_by_type[lm_type]:
            types.append(lm_type)
            means.append(np.mean(landmarks_by_type[lm_type]))
            stds.append(np.std(landmarks_by_type[lm_type]))
    
    x_pos = np.arange(len(types))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ax4.bar(x_pos, means, yerr=stds, capsize=5, color=colors[:len(types)], alpha=0.7, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(types)
    ax4.set_title("Mean Error by Landmark Type (with Std Dev)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Landmark Type", fontsize=10)
    ax4.set_ylabel("Mean Error (mm)", fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # 6. Violin plot by specific type
    ax5 = plt.subplot(2, 3, 5)
    data_by_type = []
    type_labels = []
    for lm_type in ['O', 'MB', 'DB', 'CB', 'CL']:
        if landmarks_by_type[lm_type]:
            data_by_type.extend(landmarks_by_type[lm_type])
            type_labels.extend([lm_type] * len(landmarks_by_type[lm_type]))
    
    type_data = pd.DataFrame({'error': data_by_type, 'type': type_labels})
    ax5 = sns.violinplot(x="type", y="error", data=type_data, ax=ax5, palette="Set2")
    ax5.set_title("Error Distribution by Specific Type", fontsize=12, fontweight='bold')
    ax5.set_xlabel("Landmark Type", fontsize=10)
    ax5.set_ylabel("Error (mm)", fontsize=10)
    
    # 7. Statistics summary as text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "STATISTICS SUMMARY\n" + "="*40 + "\n\n"
    summary_text += f"Total Landmarks: {len(df['error'])}\n"
    summary_text += f"Mean Error: {np.mean(df['error']):.4f} mm\n"
    summary_text += f"Std Dev: {np.std(df['error']):.4f} mm\n"
    summary_text += f"Min Error: {np.min(df['error']):.4f} mm\n"
    summary_text += f"Max Error: {np.max(df['error']):.4f} mm\n"
    summary_text += f"Median Error: {np.median(df['error']):.4f} mm\n\n"
    
    success_count = np.sum(np.array(df['error']) < 5)
    fail_count = np.sum(np.array(df['error']) >= 5)
    summary_text += f"Success (<5mm): {success_count} ({success_count/len(df['error'])*100:.1f}%)\n"
    summary_text += f"Fail (≥5mm): {fail_count} ({fail_count/len(df['error'])*100:.1f}%)\n\n"
    
    summary_text += "By Type:\n" + "-"*40 + "\n"
    for lm_type in ['O', 'MB', 'DB', 'CB', 'CL']:
        if landmarks_by_type[lm_type]:
            count = len(landmarks_by_type[lm_type])
            mean_err = np.mean(landmarks_by_type[lm_type])
            summary_text += f"{lm_type}: {count} landmarks, Mean: {mean_err:.4f}mm\n"
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def remove_extra_faces(F,num_faces,RI,label):
    last_num_faces =[]
    # print(num_faces)
    # print(RI)
    # print(len(num_faces))
    for face in num_faces:
        # print('label :',label)
        # print('face :',face.item())
        # print('RI.shape :',RI.shape)
        # print(RI.squeeze(0)[int(face.item())])
        # print(F.shape)
        # print(F.squeeze(0)[int(face.item())])
        vertex_color = F.squeeze(0)[int(face.item())]
        # print(vertex_color)
        for vert in vertex_color:
            # print("vert :",vert)
            # print(RI.squeeze(0)[vert])
            if RI.squeeze(0)[vert] == label:
                last_num_faces.append(face)
            # else:
            #     print('wrong label')
    return last_num_faces
