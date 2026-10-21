import pytest
from tools.prepare_visdrone import partition_groups
from feddecouple.data import read_labels

def test_group_partition_conservation():
    groups={str(i):[{'objects':[{'category_id':i%10+1}]}]*(i+1) for i in range(15)}
    clients,counts,target=partition_groups(groups,5,.1,42)
    assert all(clients)
    assert sum(map(len,clients))==sum(map(len,groups.values()))
    again=partition_groups(groups,5,.1,42)
    assert counts==again[1]

def test_visdrone_mapping(tmp_path):
    p=tmp_path/'a.txt'
    p.write_text('0,0,10,10,1,1,0,0\n20,20,10,10,0,0,0,0\n30,30,10,10,1,11,0,0\n')
    obj,ignore=read_labels(p,100,100)
    assert len(obj)==1 and obj[0]['category_id']==1 and len(ignore)==2
