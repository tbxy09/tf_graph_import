import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
# %load src/graph_of.py
from lucid.misc.io.reading import read


def sess_pp():

    import tensorflow as tf

    sess=tf.InteractiveSession()
    init=tf.global_variables_initializer()
    sess.run(init)
    return sess

def placeholder():
    import dill

    # with open('./data/vars/checkpoint.pkl','rb') as f:
        # ph=dill.load(f)
    # ph=dill.load(open('./data/vars/checkpoint.pkl','rb'))

    with open('./data/vars/checkpoint_var.pkl','rb') as f:
        values=dill.load(f)
    # ph=dill.load(open('./data/vars/checkpoint.pkl','rb'))
    ph=[tf.placeholder(tf.as_dtype(each.dtype),shape=each.shape) for each in values]
    feed={p:inp for p,inp in zip(ph,values)}
    # with open('./data/vars/checkpoint_feed.pkl','rb') as f:
        # feed=dill.load(f)

    return values,ph,feed

def restore(inp, tensor, tf):
    with tf.Session(config=tf.ConfigProto(device_count={'cpu': 0})) as sess:
        inp.restore(sess)
        # b=tf.shape(inp['hits'])
        print(sess.run(tf.report_uninitialized_variables()))
        # inp.hits.eval().shape
        print(sess.run([tensor, inp.hits]))

def get_vars():
    # freeze a method,then to create local vars

    import dill
    f=open('./data/vars/feeder_meta.pkl','rb')
    inp=dill.load(f)

    inp_vars=inp._var_dict(inp._build_vars())

    return inp_vars

# %load src/wgd.py
def pack_graph_def(tfnode, name,out):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # graph=convert_variables_to_constants(sess,tfnode.graph.as_graph_def(),[out])
        tf.train.write_graph(
            # tfnode.graph.as_graph_def(),
            graph,
            './data/vars',
            f'{name}.pb',
            as_text=False)

    # return tfnode.graph.as_graph_def()


# def pack_graph_def(tfnode, name):
#     # sess.run(tf.global_variables_initializer())
#     # graph=convert_variables_to_constants(sess,tfnode.graph.as_graph_def(),[out])
#     tf.train.write_graph(
#         tfnode.graph.as_graph_def(),
#         # graph,
#         './data/vars',
#         f'{name}.pb',
#         as_text=False)

#     return tfnode.graph.as_graph_def()
# unpack the graph_def
def unpack_graph_def(f):

    the_string = read(f'./data/vars/{f}.pb')
    graph_def = tf.GraphDef.FromString(the_string)
    output = tf.import_graph_def(graph_def, return_elements=['out:0'])
    return graph_def



# unpack the graph_def
def unpack_graphdef_txt():

    # the code,paste from https://tang.su/2017/01/export-TensorFlow-network/
    from google.protobuf import text_format

    with tf.Session() as sess:
        with open('./graph.pb', 'r') as f:
            graph_def = tf.GraphDef()
            text_format.Merge(f.read(), graph_def)
            output = tf.import_graph_def(graph_def, return_elements=['out:0'])
            print(sess.run(output))


def unpack_graphdef():

    # the code,paste from https://tang.su/2017/01/export-TensorFlow-network/
    from google.protobuf import text_format

    with tf.Session() as sess:
        with open('./graph.pb', 'r') as f:
            text_format.Merge(f.read(), graph_def)
            output = tf.import_graph_def(graph_def, return_elements=['out:0'])
            print(sess.run(output))
# %load src/dl.py
def dl(var,name):

    import dill

    with open(f'./data/vars/checkpoint_{name}.pkl', 'wb') as f:
        dill.dump(var, f)


def ll(var):

    import dill

    with open(f'./data/vars/checkpoint_{var}.pkl', 'rb') as f:
        return dill.load(f)
# the_string=read('/opt/playground/databowl.pb')
# the_string=read('./data/vars/sess_graph.pb')


def graph_of(f):
    the_string = read(f'./data/vars/{f}.pb')
    graph_def = tf.GraphDef.FromString(the_string)

    return graph_def


def import_graph(graph_def, t_input=None, scope='import', forget_xy_shape=True):
    """Import model GraphDef into the current graph."""
    graph = tf.get_default_graph()
    assert graph.unique_name(scope, False) == scope, (
        'Scope "%s" already exists. Provide explicit scope names when '
        'importing multiple instances of the model.') % scope
    t_input, t_prep_input = self.create_input(t_input, forget_xy_shape)
    tf.import_graph_def(
        graph_def, {self.input_name: t_prep_input}, name=scope)
    self.post_import(scope)


# %load -n import_graph
def import_graph(name):
    sess=tf.InteractiveSession()
    # graph_def=graph_of('tfnode_timex')
    graph_def=graph_of(name)
    tf.import_graph_def(graph_def=graph_def,name='xy')
    return sess

class GraphDef():
    def __init__(self,path=None):
        if path==None:
            self.from_sess()
            self.show(im=0)
        else:
            self.path=path
            self.graph_def=graph_of(path)
        # self.from_sess()
        # self.__repr__()
    # (graph_def,from_session,from_tfnode):
        # if from_session:
        #     self.graph_def=sess.graph_def
        # else if from_tfnode:
        #     self.graph_def=tfnode.graph.as_graph_def()
        # else:
        #     self.graph_def=graph_def

    def from_sess(self,sess=None):
        if sess==None:
            with tf.Session() as sess:
                self.graph_def=sess.graph_def
                print(self.shape())
                return 1
        self.graph_def=sess.graph_def
        return 1
    # def convert_const(self,tensor,init_op):
    def convert_const(self,tensor,p):
        def decorator(init_op):
            with tf.Session() as sess:
                init_op(p,sess)
                graph=convert_variables_to_constants(sess,sess.graph_def,[tensor])

                print(self.shape())
                # self.get_tensor('import/hits:0')
                self.graph_def=graph
                self.__repr__()
                # self.get_tensor('import/hits:0')
                print(self.shape())

                tf.train.write_graph(graph,'.','./data/vars/time_x.pb',as_text=False)
        return decorator

    def from_tfnode(self,tfnode):
        self.graph_def=tfnode.graph.as_graph_def()

    def get_tensor(self,name):
        with tf.Graph().as_default(),tf.Session() as sess:

            tf.import_graph_def(self.graph_def,name='import')
            return sess.graph.get_tensor_by_name(name)

    def show(self,im):
        if(im==1):

            with tf.Graph().as_default(),tf.Session() as sess:
                # tf.reset_default_graph()

                tf.import_graph_def(self.graph_def,name='import')
                print(sess.graph.get_all_collection_keys(),
                            # sess.graph.get_collection('variables'),
                            sess.graph.get_collection('local_variables'))

        else:
            with tf.Session() as sess:

                print(sess.graph.get_all_collection_keys(),
                            # sess.graph.get_collection('variables'),
                            sess.graph.get_collection('local_variables'))



    def list_tensor(self,name):
        [print(each.name) for each in self.graph_def.node if name in each.name]


    def shape(self):
        return len(self.graph_def.node)

    def imp(self):
        tf.import_graph_def(self.graph_def,name='import')


