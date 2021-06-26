# coding:utf-8

# Huffman Coding #
# from: https://www.programmersought.com/article/27913639346/

#Tree-Node Type
class Node:
    def __init__(self,freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq
    def isLeft(self):
        return self.father.left == self
#create nodesCreate leaf nodes
def createNodes(freqs):
    return [Node(freq) for freq in freqs]

#create Huffman-TreeCreate Huffman-Tree
def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item:item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]
#Huffman coding
def huffmanEncoding(nodes,root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes

# Unzip the huffman file
def decode_huffman(input_string,  char_store, freq_store):
    #input_string Huffman encoding
    #char_store character set 
    #freq_store Character transcoding 01 sequence
    encode = ''
    decode = ''
    for index in range(len(input_string)):
        encode = encode + input_string[index]
        for item in zip(char_store, freq_store):
            if encode == item[1]:
                decode = decode + item[0]
                encode = ''
    return decode;           

#Get Huffman encoding
def getHuffmanCode(string):   
    dict1 ={}
    for  i in string:
        if i in dict1.keys():
            dict1[i] += 1
        else :
            dict1[i] = 1 
    #Sort characters according to frequency
    chars_freqs  = sorted(dict1.items(), key = lambda kv:(kv[1], kv[0]))
    #Create huffman node tree
    nodes = createNodes([item[1] for item in chars_freqs])
    root = createHuffmanTree(nodes)
    #Huffman encoding of each character
    codes = huffmanEncoding(nodes,root)
    #print codes
    dict2 = {}
    for item in zip(chars_freqs,codes):
        #print 'Character:%s freq:%-2d   encoding: %s' % (item[0][0],item[0][1],item[1])
        dict2[item[0][0]] = item[1]
    str = ''
    for v in string:
        str += dict2[v]
    return [str,dict2]

# usage
#mask = map(str, mask)
#mask = ','.join(mask)
#comp_mask = getHuffmanCode(mask)
#a = decode_huffman(a, b.keys(), b.values()) #ohne comma :D
#np.fromstring(a, dtype=int, sep='')
        
#error = map(str, error)
#error = ','.join(error)
#comp_error = getHuffmanCode(error)