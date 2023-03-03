This repository contains our work<br />
**Graph Neural Networks for Distributed Power Allocation in Wireless Networks: Aggregation Over-the-Air**, which is available at https://arxiv.org/abs/2207.08498, and to be appear in TWC.<br />

**For any reproduce, further research or development, please kindly cite our paper**<br />
@ARTICLE{GNN_aggregation_OTA,<br />
       author = {{Gu}, Yifan and {She}, Changyang and {Quan}, Zhi and {Qiu}, Chen and {Xu}, Xiaodong},<br />
        title = "{Distributed Graph Neural Networks for Optimizing Wireless Networks: Message Passing Over-the-Air}",<br />
      journal = {arXiv e-prints},<br />
         year = 2022,<br />
        month = jul,<br />>
archivePrefix = {arXiv},<br />
       eprint = {2207.08498},<br />
 primaryClass = {eess.SP},<br />
}<br />

**Instructions:**<br />
1. Simulation for MPNN, WMMSE and EPA policies can be found in **MPNN and WMMSE and EPA.py**.<br />
2. Simulation for the proposed Air-MPNN can be found in **Air-MPNN.py**.<br />
3. Simulation for the proposed Air-MPRNN can be found in **Air-MPRNN.py**.<br />
4. We give examples for scalability and signaling overhead simulations.<br />
   To consider different link densities for testing, change the parameter **filed_length** in the line **test_config.field_length = field_length**.<br />
   To consider different channel correlation coefficient for testing, change the parameter **r** in the helper_functions.py.<br />
   
We thank the works "Graph Neural Networks for Scalable Radio Resource Management: Architecture Design and Theoretical Analysis" and "Spatial Deep Learning for Wireless Scheduling" for their source codes in creating this repository.
