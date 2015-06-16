from control3.ale_mdps import *
for key in ALE_METADATA:
    print "checking",key
    mdp = get_mdp("ale:"+key)
    print "ok"
