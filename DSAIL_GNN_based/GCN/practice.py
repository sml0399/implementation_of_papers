import os
root_path=os.path.abspath(__file__)
while(1):
    if(root_path[-24:]=="implementation_of_papers"):
        break
    else:
        root_path=os.path.dirname(root_path)
print(root_path)