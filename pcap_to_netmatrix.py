"""
LiM – 将 PCAP 会话转换为 NetMatrix 表格表示

用途概览：
- 输入：按“类别/会话.pcap”组织的数据集目录（每个会话是一个 PCAP 文件）
- 处理：为每个会话提取前 N 个包含加密应用数据的 TLS 数据包，计算三类特征
  - `ip_total_len`：IP 层总长度
  - `ip_ttl`：IP 层 TTL（生存时间）
  - `inter_arrival_time`：相邻数据包到达时间间隔
- 输出：将所有会话的特征行合并为 CSV（NetMatrix），文件名为
  `<数据集父目录名>_<N>_packets.csv`

核心思想：依据 RFC 的协议语义，仅使用轻量、可解释的头部/时序特征来进行加密流量分类，避免深度学习的复杂度。
"""

import os
import pyshark
from tqdm import tqdm
import pandas as pd

# Parent path of the dataset which includes traffic session pcaps of classes as subfolders 
# Dataset should be in the following format:
# dataset_path/
# ├── class1/
# │   ├── session1.pcap
# │   ├── session2.pcap
# │   └── ...
# ├── class2/
# │   ├── session1.pcap
# │   ├── session2.pcap
# │   └── ...
# └── ...
# 数据集父目录（需要替换为你自己的路径）。
# 目录结构示例：
# dataset_path/
# ├── class1/  # 类别名称作为标签
# │   ├── session1.pcap
# │   ├── session2.pcap
# ├── class2/
# │   ├── session1.pcap
# │   ├── session2.pcap
# └── ...
dataset_path = 'path/to/the/parent/folder/of/the/dataset/'


# 每个会话只选取前 5 个包含加密应用数据的 TLS 数据包
no_of_packets_per_session = 5


def create_dataframe(no_of_packets_per_session):
    """
    创建一个用于存放特征的 DataFrame。

    每个会话会生成 `5 * 3` 个特征，即15个特征：
    - 每个数据包三列：`<pN>_ip_total_len`, `<pN>_ip_ttl`, `<pN>_inter_arrival_time` 
    - 总共 5 个数据包，一共 15 列
    - 15列特征再加上一列 `label`（类别名，来自父目录名）。
    """
    columns = []
    # 遍历这个会话，生成特征行，每个会话生成 15 个特征
    for i in range(no_of_packets_per_session):
        packet_number = 'p' + str(i + 1)
        column_temp = [
            packet_number + '_ip_total_len', 
            packet_number + '_ip_ttl', 
            packet_number + '_inter_arrival_time'
        ]
        columns.extend(column_temp)
    columns.append('label')
    df = pd.DataFrame(columns=columns)
    return df


def get_packets_with_encrypted_application_data(pcap_file_path, no_of_packets_per_session):
    """
    使用 Pyshark 读取 PCAP,选取TLS协议加密包含应用数据的包,每个会话选择5个,选够了就行

    选择策略：
    - 仅保留 TLS应用数据包 tls_app_data,不保留TLS握手包 tls_handshake。
    - 最多采集 `no_of_packets_per_session` 5个加密应用数据包,多余的忽略。
    """
    packets_with_encrypted_payload = []
    try:
        # 使用 Wireshark/TShark 语法的显示过滤器，仅解析 TLS 层
        with pyshark.FileCapture(pcap_file_path, display_filter='tls') as capture:
            for packet in capture:
                try:
                    # 判断是否存在 TLS 层，且为应用数据（不含握手）
                    if hasattr(packet, 'tls'):
                        if hasattr(packet.tls, 'app_data') and not hasattr(packet.tls, 'handshake'):
                            packets_with_encrypted_payload.append(packet)
                    # 收集到足够数量即停止
                    if len(packets_with_encrypted_payload) >= no_of_packets_per_session:
                        break
                except Exception as packet_error:
                    print(f"Error processing packet: {packet_error}")
                    continue
    except Exception as capture_error:
        print(f"Error opening capture file: {capture_error}")
    
    return packets_with_encrypted_payload


# 生成空的特征文件表，只有15特征+1标签的表头
df = create_dataframe(no_of_packets_per_session)

# 遍历数据集，在每个会话session中读取5个TLS加密应用数据包
# 然后遍历5个应用数据包，每个应用数据包提取total_len,ttl,inter_arrival_time三个特征
# 最后将所有特征行合并为 CSV 文件，文件名为 `<数据集父目录名>_5_packets.csv`
for parent, _, files in os.walk(dataset_path):
    if files:
        # 使用父目录名作为标签（跨平台兼容）
        label = os.path.basename(parent)
        print(f'Label - {label}')

        for file in tqdm(files):
            try:
                pcap_file_path = os.path.join(parent, file)
                session = []

                packets_with_application_data = get_packets_with_encrypted_application_data(
                    pcap_file_path, no_of_packets_per_session
                )

                if len(packets_with_application_data) == no_of_packets_per_session:
                    for i, packet in enumerate(packets_with_application_data):
                        ip_total_len = None
                        ip_ttl = None
                        
                        # IP 层可能不存在，需判空
                        if hasattr(packet, 'ip'):
                            # Pyshark 字段通常是字符串，这里转换为 int
                            ip_total_len = int(packet.ip.len) if hasattr(packet.ip, 'len') else None
                            ip_ttl = int(packet.ip.ttl) if hasattr(packet.ip, 'ttl') else None

                        # 使用抓包时间戳计算相邻数据包的到达时间差
                        if i == 0:
                            inter_arrival_time = 0
                        else:
                            # 将时间戳转换为 float 后相减得到间隔
                            inter_arrival_time = float(packet.sniff_timestamp) - float(packets_with_application_data[i - 1].sniff_timestamp)
            
                        packet_features = [ip_total_len, ip_ttl, inter_arrival_time]
                        session.extend(packet_features)

                    session.append(label)
                    df.loc[len(df)] = session
                else:
                    # 未达到所需的加密应用数据包数量，跳过该会话
                    print(f"{pcap_file_path} : Less than {no_of_packets_per_session} encrypted packets found")
            
            except Exception as e:
                print(e)
                print(f"{pcap_file_path} : Error in extracting features")
                continue

# 规范化路径，确保以分隔符结尾（跨平台）
dataset_path = os.path.join(dataset_path, '')

# 输出文件名以父目录名加上数据包数拼接，便于追溯来源
parent_folder_name = os.path.basename(os.path.normpath(dataset_path))
netmatrix_file_name = f"{parent_folder_name}_{no_of_packets_per_session}_packets.csv"
output_path = os.path.join('./', netmatrix_file_name)
df.to_csv(output_path, index=False)

print(f"Netmatrix file saved at {output_path}")
