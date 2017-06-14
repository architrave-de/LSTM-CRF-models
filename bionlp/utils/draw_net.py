#### Taken from: https://gist.github.com/ebenolson/1682625dc9823e27d771
#### Requires globally installed graphviz and Python package pydot3
# sudo apt-get install graphviz
# pip install pydot3

"""
Functions to create network diagrams from a list of Layers.

Examples:

    Draw a minimal diagram to a pdf file:
        layers = lasagne.layers.get_all_layers(output_layer)
        draw_to_file(layers, 'network.pdf', output_shape=False)

    Draw a verbose diagram in an IPython notebook:
        from IPython.display import Image #needed to render in notebook

        layers = lasagne.layers.get_all_layers(output_layer)
        dot = get_pydot_graph(layers, verbose=True)
        return Image(dot.create_png())
"""

import lasagne
import pydot

from bionlp.taggers.rnn_feature.networks.crf_dual_layer import DualCRFLayer

def get_general_attributes():
    return ['name', 'shape']

def get_class_attributes(layer):
    if isinstance(layer, lasagne.layers.EmbeddingLayer):
        return ['input_size', 'output_size']
    if isinstance(layer, lasagne.layers.DenseLayer):
        return ['num_units', 'nonlinearity', 'num_leading_axes']
    if isinstance(layer, lasagne.layers.ConcatLayer):
        return ['axis', 'cropping']
    if isinstance(layer, lasagne.layers.DropoutLayer):
        return ['p', 'rescale', 'shared_axes']
    if isinstance(layer, lasagne.layers.LSTMLayer):
        return ['num_units', 'nonlinearity', 'backwards', 'learn_init',
        'gradient_steps', 'grad_clipping', 'unroll_scan', 'only_return_final']
    if isinstance(layer, lasagne.layers.BatchNormLayer):
        return ['axes', 'epsilon', 'alpha']
    if isinstance(layer, lasagne.layers.NonlinearityLayer):
        return ['nonlinearity']
    if isinstance(layer, lasagne.layers.DimshuffleLayer):
        return ['pattern']
    if isinstance(layer, lasagne.layers.Conv1DLayer):
        return ['num_filters', 'filter_size', 'stride', 'pad',
        'untie_biases', 'nonlinearity', 'flip_filters', 'convolution']
    if isinstance(layer, DualCRFLayer):
        return ['mask_input']
    return []

def get_hex_color(layer_classname):
    """
    Determines the hex color for a layer. Some classes are given
    default values, all others are calculated pseudorandomly
    from their name.
    :parameters:
        - layer_classname : string
            Class name of the layer

    :returns:
        - color : string containing a hex color.

    :usage:
        >>> color = get_hex_color('MaxPool2DDNN')
        '#9D9DD2'
    """

    if 'Input' in layer_classname:
        return '#A2CECE'
    if 'Conv' in layer_classname:
        return '#7C9ABB'
    if 'Dense' in layer_classname:
        return '#6CCF8D'
    if 'Pool' in layer_classname:
        return '#9D9DD2'
    else:
        # create a color from the hash of the class name, but make
        # sure that it is relatively bright, so add half of the class
        # name hash to #7F7F7F
        layer_name_hash = (hash(layer_classname) % 2**24) / 2
        return "#{0:x}".format(int(layer_name_hash + float.fromhex("7F7F7F")))


def get_pydot_graph(final_layer, output_shape=True, verbose=False):
    """
    Creates a PyDot graph of the network defined by the layers retrieved
    from the given final_layer.
    :parameters:
        - final_layer : lasagne.layers.base.Layer
            Final layer of the neural net, from which a list of all
            layers is obtained via lasange.layers.get_all_layers
        - output_shape: (default `True`)
            If `True`, the output shape of each layer will be displayed.
        - verbose: (default `False`)
            If `True`, layer attributes like filter shape, stride, etc.
            will be displayed.
        - verbose:
    :returns:
        - pydot_graph : PyDot object containing the graph

    """
    layers = lasagne.layers.get_all_layers(final_layer)
    pydot_graph = pydot.Dot('Network', graph_type='digraph')
    pydot_nodes = {}
    pydot_edges = []
    for i, layer in enumerate(layers):
        layer_classname = '{0}'.format(layer.__class__.__name__)
        key = repr(layer)
        label = layer_classname
        color = get_hex_color(layer_classname)
        if verbose:
            general_attributes = get_general_attributes()
            class_attributes = get_class_attributes(layer)
            for attr_name in (general_attributes + class_attributes):
                if hasattr(layer, attr_name):
                    attr_value = getattr(layer, attr_name)
                    # the names of callables need to be read in a different way
                    if hasattr(attr_value, "__call__"): # Python 3 style of checking for callables; see: https://stackoverflow.com/a/2435074/2191154
                        try:
                            callable_name= attr_value.__name__
                        except AttributeError:
                            callable_name = attr_value.__class__.__name__
                        label += '\n' + '{0}: {1}'.format(attr_name, callable_name)
                    else:
                        label += '\n' + '{0}: {1}'.format(attr_name, attr_value)

        if output_shape:
            label += '\n' + \
                'Output shape: {0}'.format(layer.get_output_shape())
        pydot_nodes[key] = pydot.Node(key,
                                      label=label,
                                      shape='record',
                                      fillcolor=color,
                                      style='filled',
                                      )

        if hasattr(layer, 'input_layers'):
            for input_layer in layer.input_layers:
                pydot_edges.append([repr(input_layer), key])

        if hasattr(layer, 'input_layer'):
            pydot_edges.append([repr(layer.input_layer), key])

    for node in pydot_nodes.values():
        pydot_graph.add_node(node)
    for edge in pydot_edges:
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edge[0]], pydot_nodes[edge[1]]))
    return pydot_graph


def draw_to_file(final_layer, filename, **kwargs):
    """
    Draws a network diagram to a file
    :parameters:
        - final_layer : lasagne.layers.base.Layer
            Final layer of the neural net, from which a list of all
            layers is obtained via lasange.layers.get_all_layers
        - filename: string
            The filename to save output to.
        - **kwargs: see docstring of get_pydot_graph for other options
    """
    layers = lasagne.layers.get_all_layers(final_layer)
    dot = get_pydot_graph(layers, **kwargs)
    
    ext = filename[filename.rfind('.') + 1:]
    with open(filename, 'wb') as fid:
        fid.write(dot.create(format=ext))


def draw_to_notebook(final_layer, **kwargs):
    """
    Draws a network diagram in an IPython notebook
    :parameters:
        - final_layer : lasagne.layers.base.Layer
            Final layer of the neural net, from which a list of all
            layers is obtained via lasange.layers.get_all_layers
        - **kwargs: see docstring of get_pydot_graph for other options
    """
    from IPython.display import Image  # needed to render in notebook

    layers = lasagne.layers.get_all_layers(final_layer)
    dot = get_pydot_graph(layers, **kwargs)
    return Image(dot.create_png())