import mjcpy2

def get_names(model, name):
	names = {}
	jntadr = model[name].squeeze()
	for i in range(len(jntadr)):
		# look for NULL terminator
		j = jntadr[i]
		while model['names'][j] != 0:
			j += 1
		names[i] = "".join(map(chr, model['names'][jntadr[i]:j]))
	return names

def get_joint_names(model):
	return get_names(model, 'name_jntadr')

def get_body_names(model):
	return get_names(model, 'name_bodyadr')

def get_geom_names(model):
	return get_names(model, 'name_geomadr')


def get_actuators(model):
	return model['actuator_trnid'][:,0]

def extract_feature(world, f, feature_name):
    startidx = 0
    for (name,size) in world.GetFeatDesc():
        if name == feature_name:
            return f[startidx:startidx+size]
            break
        else:
            startidx += size