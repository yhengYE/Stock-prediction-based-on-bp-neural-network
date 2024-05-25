# -*- coding:utf-8 -*-
import argparse
import tkinter as tk
import tkinter.font as tkFont
default_args = {
    'hidden_neurons_1': 100,
    'hidden_neurons_2': 8,
    'batch_size': 10,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,

}
args = {}
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)        #训练次数
    parser.add_argument('--learning_rate', type=float, default=0.001)  #学习率
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--activation_function',default='relu')
    parser.add_argument('--lr_change', default='YES', choices=('YES', 'NO'))#优化器
    parser.add_argument('--hidden_neurons_1', default='100')
    parser.add_argument('--hidden_neurons_2', default='8')
    parser.add_argument('--value', default="'Last'")
    args = parser.parse_args()
    return args
def fill_defaults(hidden_layer1_entry, hidden_layer2_entry, batch_size_entry, epochs_entry, learning_rate_entry, weight_decay_entry):
    # 填充默认参数值到GUI输入框
    hidden_layer1_entry.delete(0, tk.END)
    hidden_layer1_entry.insert(0, str(default_args['hidden_neurons_1']))

    hidden_layer2_entry.delete(0, tk.END)
    hidden_layer2_entry.insert(0, str(default_args['hidden_neurons_2']))

    batch_size_entry.delete(0, tk.END)
    batch_size_entry.insert(0, str(default_args['batch_size']))

    epochs_entry.delete(0, tk.END)
    epochs_entry.insert(0, str(default_args['epochs']))

    learning_rate_entry.delete(0, tk.END)
    learning_rate_entry.insert(0, str(default_args['learning_rate']))

    weight_decay_entry.delete(0, tk.END)
    weight_decay_entry.insert(0, str(default_args['weight_decay']))

def get_gui_args():
    def submit():
        args['hidden_neurons_1'] = int(hidden_layer1_entry.get())
        args['hidden_neurons_2'] = int(hidden_layer2_entry.get())
        args['batch_size'] = int(batch_size_entry.get())
        args['epochs'] = int(epochs_entry.get())
        args['learning_rate'] = float(learning_rate_entry.get())
        args['weight_decay'] = float(weight_decay_entry.get())
        args['activation_function'] = activation_func_var.get()
        args['value'] = value_var.get()

        root.quit()

    # 创建窗口
    root = tk.Tk()
    root.geometry("450x500")
    root.title("训练参数")
    args = {}


    # 设置网格的行配置，为组件之间添加空间
    for row_index in range(12):
        root.grid_rowconfigure(row_index, weight=1)
    # 定义更大的字体
    large_font = tkFont.Font(size=16)
    button_font = tkFont.Font(size=16)

    mywidth=20;



    title_frame = tk.LabelFrame(root, text="", padx=10, pady=10)  # padx 和 pady 分别是内部填充
    title_frame.grid(row=0, column=0, columnspan=2, padx=50, pady=10, sticky="ns")

    # 在 LabelFrame 内部创建标签，不需要再设置 columnspan，因为它已经在 LabelFrame 中
    title_label = tk.Label(title_frame, text="基于BP神经网络的股票价格预测", font=large_font,)
    title_label.pack(anchor='center')  # 使用 pack() 布局管理器来放置标签


    # 创建并放置组件
    large_font = tkFont.Font(size=16)
    value_var = tk.StringVar(root)
    value_var.set("Last")  # 设置默认值
    value_var_menu = tk.OptionMenu(root, value_var, "High","Low","Last")
    value_var_menu.config(font=large_font,width=8)
    value_var_menu.grid(row=1, column=0, padx=(55, 10), sticky='ns')

    fill_defaults_button = tk.Button(root, text="一键输入", font=button_font, height=1, width=10,  command=lambda: fill_defaults(hidden_layer1_entry,hidden_layer2_entry, batch_size_entry,  epochs_entry, learning_rate_entry,weight_decay_entry))
    fill_defaults_button.grid(row=1, column=1, padx=(10, 5), sticky='ns')



    hidden_layer1_entry = tk.Entry(root, font=large_font, width=mywidth)
    tk.Label(root, text="隐藏层1:", font=large_font).grid(row=2, column=0,padx=(30, 0))
    hidden_layer1_entry.grid(row=2, column=1)


    hidden_layer2_entry = tk.Entry(root, font=large_font, width=mywidth)
    tk.Label(root, text="隐藏层2:", font=large_font).grid(row=3, column=0,padx=(30, 0))
    hidden_layer2_entry.grid(row=3, column=1)


    tk.Label(root, text="训练批量:", font=large_font).grid(row=4, column=0,padx=(30, 0))
    batch_size_entry = tk.Entry(root, font=large_font, width=mywidth)  # 假设原宽度为10
    batch_size_entry.grid(row=4, column=1)

    tk.Label(root, text="迭代次数:", font=large_font).grid(row=5, column=0,padx=(30, 0))
    epochs_entry = tk.Entry(root, font=large_font, width=mywidth)
    epochs_entry.grid(row=5, column=1)

    tk.Label(root, text="学习率:", font=large_font).grid(row=6, column=0,padx=(30, 0))
    learning_rate_entry = tk.Entry(root, font=large_font, width=mywidth)
    learning_rate_entry.grid(row=6, column=1)

    tk.Label(root, text="权重衰减:", font=large_font).grid(row=7, column=0,padx=(30, 0))
    weight_decay_entry = tk.Entry(root, font=large_font, width=mywidth)
    weight_decay_entry.grid(row=7, column=1)

    large_font = tkFont.Font(size=16)
    tk.Label(root, text="激活函数:", font=large_font).grid(row=8, column=0,padx=(30, 0))
    activation_func_var = tk.StringVar(root)
    activation_func_var.set("ReLU")  # 设置默认值
    activation_func_menu = tk.OptionMenu(root, activation_func_var, "ReLU", "Swish")
    activation_func_menu.config(font=large_font)
    activation_func_menu.grid(row=8, column=1)


    submit_button = tk.Button(root, text="开始训练", font=button_font, height=1, width=10,command=submit)  # 假设原高度为1，宽度为10
    submit_button.grid(row=9, column=0, columnspan=2)

    # 启动主循环
    root.mainloop()
    root.destroy()
    return args

if __name__ == "__main__":
    get_args()
    training_args = get_gui_args()
    print(training_args)