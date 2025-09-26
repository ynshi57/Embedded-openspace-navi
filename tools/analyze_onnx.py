import onnx
import onnxruntime as ort
import numpy as np
import os
import sys
import onnx.numpy_helper as numpy_helper
import matplotlib.pyplot as plt
import argparse
import pandas as pd
# parser = argparse.ArgumentParser(description='Process an ONNX model file.')
# parser.add_argument('onnx_model_path', type=str, help='The path to the ONNX model file')
# args = parser.parse_args()
# onnx_model_path = args.onnx_model_path
# onnx_model = onnx.load(onnx_model_path)
# # onnx_model = onnx.load("./hm_6v_od_s8_res34_simple.onnx")
# onnx_model = onnx.load("./hm_6v_od_s8_res34_simple.onnx")
# parser = argparse.ArgumentParser(description='Process an ONNX model file.')
# parser.add_argument('onnx_model_path', type=str, help='The path to the ONNX model file')
# parser.add_argument('--nchw', type=str, help='Your new argument')  
# args = parser.parse_args()
# onnx_model_path = args.onnx_model_path
def parse_onnx(onnx_path):
    # Load the model
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    # Add intermediate outputs
    intermediate_layer_value_info = []
    for node in model.graph.node:
        for output in node.output:
            value_info = onnx.helper.ValueInfoProto()
            value_info.name = output
            intermediate_layer_value_info.append(value_info)
    model.graph.output.extend(intermediate_layer_value_info)

    # Extract input shapes
    input_name_shape_dict = [
        {
            'name': input.name, 
            'shape': [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        } 
        for input in model.graph.input
    ]
    
    return model, input_name_shape_dict

def forward_onnx(model, input_name_shape_dict, input_dir):
    # Convert the model to a binary string for onnxruntime
    model_bin = model.SerializeToString()

    # Prepare input data
    name2data = {}
    for each in input_name_shape_dict:
        name = each["name"]
        shape = each["shape"]
        file_path = os.path.join(input_dir, name + ".bin")
        if not os.path.isfile(file_path):
            raise Exception(f"Cannot find input file: {file_path}")

        data = np.fromfile(file_path, dtype=np.float32)
        if np.prod(shape) != data.size:
            raise Exception(f"Input data size ({data.size}) does not match model expected size ({np.prod(shape)}).")

        data = data.reshape(shape)
        name2data[name] = data

    # Create an inference session
    ort_sess = ort.InferenceSession(model_bin, providers=['CPUExecutionProvider'])

    # Run inference and get intermediate outputs
    model_output_names = [out.name for out in ort_sess.get_outputs()]
    onnx_outputs = ort_sess.run(model_output_names, name2data)
    save_path = './save1'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Print each output
    within_fp16_range = []
    outside_fp16_range = []
    layer_means = []
    layer_mins = []
    layer_maxs = []
    layer_vars = []
    layer_data = []
    for i, output_name in enumerate(model_output_names):
        output_data = onnx_outputs[i]
            # print(f"Output for node '{output_name}':\n{output_data}")
            #统计最小最大均值方差
        layer_means.append(np.mean(output_data))
        layer_mins.append(np.min(output_data))
        layer_maxs.append(np.max(output_data))
        # layer_vars.append(np.var(output_data))
        layer_vars.append(np.std(output_data))
        if (output_data >= -65504).all() and (output_data <= -6e-8).all() or (output_data >= 6e-8).all() and (output_data <= 65504).all() or (output_data == 0).all():

            within_fp16_range.append((output_data >= -65504).all() and  (output_data <= -6e-8).all() or (output_data >= 6e-8).all() and (output_data <= 65504).all() or (output_data == 0).all())
        else:
            outside_fp16_range.append((output_data<-65504 ).all() or (output_data > -6e-8).all() and (output_data <0).all() or (output_data >0).all() and   (output_data < 6e-8).all()  or   (output_data > 65504).all())  

        total_values = len(within_fp16_range) + len(outside_fp16_range)     #total_values = len(min_values_with_fp16_range) + len(min_values_outside_fp16_range)
        percentage_within_range = len(within_fp16_range) / total_values * 100             #percentage_within_range = len(min_values_within_fp16_range) / total_values*100
        percentage_outside_range = len(outside_fp16_range) / total_values * 100   #;len()   计算数量相加即可   
        layer_stats = {
            'Name': output_name,
            'Mean': np.mean(output_data),
            'Min': np.min(output_data),
            'Max': np.max(output_data),
            # 'Var': np.var(output_data),
            'Std': np.std(output_data),
            'within FP16': percentage_within_range,
            'outside FP16': percentage_outside_range ,
            
        }
        layer_data.append(layer_stats)
        hist_within_range, bins_within_range = np.histogram(within_fp16_range, bins=np.arange(0, 4.5, 0.5), density=True)
        # Normalize the histogram so that the probabilities sum up to 1
        hist_within_range = hist_within_range / np.sum(hist_within_range)

        # Plot the histogram for values outside the FP16 range
        hist_outside_range, bins_outside_range = np.histogram(outside_fp16_range, bins=np.arange(0, 4.5, 0.5), density=True)

        # Normalize the histogram so that the probabilities sum up to 1
        hist_outside_range = hist_outside_range / np.sum(hist_outside_range)

        # #保存统计信息到文本文件
        # stats_file_path = f"{save_path}/{output_name}_stats.txt"
        # with open(stats_file_path, "w") as f:
        #     f.write(f"Statistics for node '{output_name}':\n")
        #     f.write(f"Mean: {np.mean(output_data)}\n")
        #     f.write(f"Min: {np.min(output_data)}\n")
        #     f.write(f"Max: {np.max(output_data)}\n")
        #     f.write(f"Var: {np.var(output_data)}\n")
        #     f.write(f"percentage_within_range : {percentage_within_range}\n")
        #     f.write(f"percentage_outside_range: {percentage_outside_range }\n")
    # Histogram of Means
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 3, 1)
        plt.hist(layer_means, bins=20, color='green', alpha=0.7)
        plt.title("Histogram of Means")
        plt.xlabel("Mean Value")
        plt.ylabel("Frequency")
        # Histogram of Minimums
        plt.subplot(2, 3, 2)
        plt.hist(layer_mins, bins=20, color='blue', alpha=0.7)
        plt.title("Histogram of Min")
        plt.xlabel("Minimum Value")
        plt.ylabel("Frequency")
        # Histogram of Maximums
        plt.subplot(2, 3, 3)
        plt.hist(layer_maxs, bins=20, color='red', alpha=0.7)
        plt.title("Histogram of Max")
        plt.xlabel("Maximum Value")
        plt.ylabel("Frequency")
        # Histogram of Variances
        plt.subplot(2, 3, 4)
        plt.hist(layer_vars, bins=20, color='purple', alpha=0.7)
        plt.title("Histogram of Std")
        plt.xlabel("Standard Value")
        plt.ylabel("Frequency")

        # plt.savefig('static.png', bbox_inches='tight')
        plt.savefig(f"{save_path}/{output_name}.png", bbox_inches='tight')        # 修改保存每一次的图
        plt.close()
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(bins_within_range[:-1], hist_within_range, width=np.diff(bins_within_range), alpha=0.5, color='blue')
        plt.xlabel("Absolute Minimum Value")
        plt.ylabel("Probability Density")
        plt.title(f"Within FP16 Range (Min Value < {6e-8  })\n{percentage_within_range:.2f}%")
        plt.xticks(np.arange(0, 4, 0.5))
        plt.xlim(0, 4)
        for i in range(len(hist_within_range)):
            if bins_within_range[i] <= 4:
                plt.text(bins_within_range[i] + 0.25, hist_within_range[i], f'{hist_within_range[i]*100:.2f}%', ha='center', va='bottom')

        plt.subplot(1, 2, 2)
        plt.bar(bins_outside_range[:-1], hist_outside_range, width=np.diff(bins_outside_range), alpha=0.5, color='red')
        plt.xlabel("Absolute Minimum Value")
        plt.ylabel("Probability Density")
        plt.title(f"Outside FP16 Range (Min Value >= {6e-8  })\n{percentage_outside_range:.2f}%")
        plt.xticks(np.arange(0, 4, 0.5))
        plt.xlim(0, 4)
        plt.savefig(f"{save_path}/{output_name}_fp16.png", bbox_inches='tight')      
        plt.close()
        for i in range(len(hist_outside_range)):
            if bins_outside_range[i] <= 4:
                plt.text(bins_outside_range[i] + 0.25, hist_outside_range[i], f'{hist_outside_range[i]*100:.2f}%', ha='center', va='bottom')

        plt.tight_layout()
        # plt.show()
#保存xlsx表格
        layer_df_1= pd.DataFrame(layer_data)
        # Save to Excel
        layer_df_1.to_excel('onnx_1.xlsx', index=False)

    # df_stats = pd.DataFrame(layer_data)    #不能同名
    # excel_filename = os.path.join(save_path, 'onnx_1.xlsx')
    # df_stats.to_excel(excel_filename, index=False)




def main(onnx_path, input_dir):
    model, input_name_shape_dict = parse_onnx(onnx_path)
    forward_onnx(model, input_name_shape_dict, input_dir)


onnx_model_path= sys.argv[1]  # 第一个参数是脚本名，第二个参数是onnx_path  
onnx_model = onnx.load(onnx_model_path)
min_values_within_fp16_range_1 = []
min_values_outside_fp16_range_1 = []
layer_means_1 = []
layer_mins_1 = []
layer_maxs_1 = []
layer_vars_1 = []
layer_data_1 = []
# layer_data_2 = []
min_fp16_value = 6e-8   #6e-8
save_path_1 = './save'
if not os.path.exists(save_path_1):
    os.makedirs(save_path_1)
for init in onnx_model.graph.initializer:    #for init in onnx_model.graph.initializer
    name_1 = init.name
    # print(name)
    tensor = numpy_helper.to_array(init)
    # tensor_float = tensor.flatten
    # min = np.min(tensor_float)
    # print(tensor)
    if tensor.size == 0:
        print(f"Skipping empty tensor: {name_1}")
        continue
    # Calculate the minimum value of the parameters
    # min_value = np.min(abs(tensor))

    # Check if the minimum value is within the specified FP16 range
    if (tensor >= -65504).all() and (tensor <= -6e-8).all() or (tensor >= 6e-8).all() and (tensor <= 65504).all() or (tensor == 0).all():
        min_values_within_fp16_range_1.append((tensor >= -65504).all() and  (tensor <= -6e-8).all() or (tensor >= 6e-8).all() and (tensor <= 65504).all() or (tensor == 0).all())
    else:
        min_values_outside_fp16_range_1.append((tensor<-65504 ).all() or (tensor > -6e-8).all() and (tensor <0).all() or (tensor >0).all() and   (tensor < 6e-8).all()  or   (tensor > 65504).all())
    
    total_values = len(min_values_within_fp16_range_1) + len(min_values_outside_fp16_range_1)     #total_values = len(min_values_with_fp16_range) + len(min_values_outside_fp16_range)
    percentage_within_range_1 = len(min_values_within_fp16_range_1) / total_values * 100             #percentage_within_range = len(min_values_within_fp16_range) / total_values*100
    percentage_outside_range_1 = len(min_values_outside_fp16_range_1) / total_values * 100   #;len()   计算数量相加即可   
    layer_means_1.append(np.mean(tensor))
    layer_mins_1.append(np.min(tensor))
    layer_maxs_1.append(np.max(tensor))
    layer_vars_1.append(np.std(tensor))

    layer_stats_1= {
        'Name': name_1,
        'Mean': np.mean(tensor),
        'Min': np.min(tensor),
        'Max': np.max(tensor),
        # 'Var': np.var(tensor),
        'Std': np.std(tensor),
        'within FP16': percentage_within_range_1,
        'outside FP16': percentage_outside_range_1 ,
    }
    layer_data_1.append(layer_stats_1)

    # Calculate the minimum value of the parameters
    # min_value = np.min(abs(tensor))

    # Check if the minimum value is within the specified FP16 range
    # if min_value >= min_fp16_value and :
    #     min_values_outside_fp16_range_1.append(min_value)
    # else:
    #     min_values_within_fp16_range_1.append(min_value)

    # print(min_values_outside_fp16_range)
    # Calculate the percentage of values within and outside the FP16 range

    # layer_stats_2= {
    #     'Name': name,
    # }
    # layer_data_2.append(layer_stats_2)
    # print(percentage_within_range)
    # print(percentage_outside_range)

    # Plot the histogram for values within the FP16 range
    hist_within_range, bins_within_range = np.histogram(min_values_within_fp16_range_1, bins=np.arange(0, 4.5, 0.5), density=True)

    # Normalize the histogram so that the probabilities sum up to 1
    hist_within_range = hist_within_range / np.sum(hist_within_range)

    # Plot the histogram for values outside the FP16 range
    hist_outside_range, bins_outside_range = np.histogram(min_values_outside_fp16_range_1, bins=np.arange(0, 4.5, 0.5), density=True)

    # Normalize the histogram so that the probabilities sum up to 1
    hist_outside_range = hist_outside_range / np.sum(hist_outside_range)
    # Histogram of Means
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 3, 1)
    plt.hist(layer_means_1, bins=20, color='green', alpha=0.7)
    plt.title("Histogram of Means")
    plt.xlabel("Mean Value")
    plt.ylabel("Frequency")
    # Histogram of Minimums
    plt.subplot(2, 3, 2)
    plt.hist(layer_mins_1, bins=20, color='blue', alpha=0.7)
    plt.title("Histogram of Min")
    plt.xlabel("Minimum Value")
    plt.ylabel("Frequency")
    # Histogram of Maximums
    plt.subplot(2, 3, 3)
    plt.hist(layer_maxs_1, bins=20, color='red', alpha=0.7)
    plt.title("Histogram of Max")
    plt.xlabel("Maximum Value")
    plt.ylabel("Frequency")
    # Histogram of Variances
    plt.subplot(2, 3, 4)
    plt.hist(layer_vars_1, bins=20, color='purple', alpha=0.7)
    plt.title("Histogram of Std")
    plt.xlabel("Standard Value")
    plt.ylabel("Frequency")

    plt.savefig(f"{save_path_1}/{name_1}.png", bbox_inches='tight')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(bins_within_range[:-1], hist_within_range, width=np.diff(bins_within_range), alpha=0.5, color='blue')
    plt.xlabel("Absolute Minimum Value")
    plt.ylabel("Probability Density")
    plt.title(f"Within FP16 Range (Min Value < {min_fp16_value})\n{percentage_within_range_1:.2f}%")
    plt.xticks(np.arange(0, 4, 0.5))
    plt.xlim(0, 4)
    for i in range(len(hist_within_range)):
        if bins_within_range[i] <= 4:
            plt.text(bins_within_range[i] + 0.25, hist_within_range[i], f'{hist_within_range[i]*100:.2f}%', ha='center', va='bottom')

    plt.subplot(1, 2, 2)
    plt.bar(bins_outside_range[:-1], hist_outside_range, width=np.diff(bins_outside_range), alpha=0.5, color='red')
    plt.xlabel("Absolute Minimum Value")
    plt.ylabel("Probability Density")
    plt.title(f"Outside FP16 Range (Min Value >= {min_fp16_value})\n{percentage_outside_range_1:.2f}%")
    plt.xticks(np.arange(0, 4, 0.5))
    plt.xlim(0, 4)
    plt.savefig(f"{save_path_1}/{name_1}_fp16.png", bbox_inches='tight')
    for i in range(len(hist_outside_range)):
        if bins_outside_range[i] <= 4:
            plt.text(bins_outside_range[i] + 0.25, hist_outside_range[i], f'{hist_outside_range[i]*100:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    # plt.show()


#保存xlsx表格
layer_df = pd.DataFrame(layer_data_1)
# Save to Excel
layer_df.to_excel('onnx.xlsx', index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <onnx_path> <input_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])















