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


def get_hex_color(layer_type):
    """
    Determines the hex color for a layer. Some classes are given
    default values, all others are calculated pseudorandomly
    from their name.
    :parameters:
        - layer_type : string
            Class name of the layer

    :returns:
        - color : string containing a hex color.

    :usage:
        >>> color = get_hex_color('MaxPool2DDNN')
        '#9D9DD2'
    """

    if 'Input' in layer_type:
        return '#A2CECE'
    if 'Conv' in layer_type:
        return '#7C9ABB'
    if 'Dense' in layer_type:
        return '#6CCF8D'
    if 'Pool' in layer_type:
        return '#9D9DD2'
    else:
        # create a color from the hash of the class name, but make
        # sure that it is relatively bright, so add half of the class
        # name hash to #7F7F7F
        layer_name_hash = (hash(layer_type) % 2**24) / 2
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
        layer_type = '{0}'.format(layer.__class__.__name__)
        key = repr(layer)
        label = layer_type
        color = get_hex_color(layer_type)
        if verbose:
            for attr in ['name', 'num_filters', 'num_units', 'ds',
                         'filter_shape', 'stride', 'strides', 'p']:
                if hasattr(layer, attr) and getattr(layer, attr) is not None:
                    label += '\n' + \
                        '{0}: {1}'.format(attr, getattr(layer, attr))
            if hasattr(layer, 'nonlinearity'):
                try:
                    nonlinearity = layer.nonlinearity.__name__
                except AttributeError:
                    nonlinearity = layer.nonlinearity.__class__.__name__
                label += '\n' + 'nonlinearity: {0}'.format(nonlinearity)

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