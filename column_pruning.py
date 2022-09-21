import numpy as np

def prune_block(block):
    size = block.shape[1]
    column = np.random.randint(0,size)
    column = np.argmin(np.abs(block).sum(0))
    #block[:,column] =0
    u, s, vh = np.linalg.svd(block, full_matrices=False)
    if np.count_nonzero(s) == 4:
        val = np.nanmin(s[s!=0])
        s[s==val] = 0
        #if vh.shape != np.diag(s).shape:
        return u.dot(np.diag(s).dot(vh))


#    print(block)
    return block

def prune_columns_matrix(w, group_size, column_size, percent):
    x = np.hsplit(w, w.shape[1]/ group_size)
    outer = []
    for arr  in x:
        y = np.vsplit(arr, arr.shape[0] / column_size)
        result = []
        for block in y:
            result.append(prune_block(block))
        outer.append(np.vstack(result))
    return np.hstack(outer)
    #print(w)


def column_prune_weights(w, group_size, column_size, percent, debug = False):
    weight_matrix = w.reshape(w.shape[0], -1)
    if debug:
        print("pre \n", weight_matrix)

    pad_amount=0
    # Pad weights to match group size                                                                                                                                    
    if weight_matrix.shape[1] % group_size != 0:
        pad_amount = group_size - weight_matrix.shape[1] % group_size
        pad = np.zeros((weight_matrix.shape[0], pad_amount))
        #weight_matrix = np.concatenate((weight_matrix, pad), axis = 1)

    pruned_matrix =  prune_columns_matrix(weight_matrix, group_size, column_size, percent)
    # remove_padding                                                                                                                                                        
    #if pad_amount > 0:
        #weight_matrix  = weight_matrix[:,0:weight_matrix.shape[1]-pad_amount]

    if debug:
        print("post ", pruned_matrix)
    return pruned_matrix.reshape(w.shape)
    
if __name__ == "__main__":
    w = np.random.randint(5, size=(4,4))#np.random.randn(4,4)
    #print(w)
    #prune_columns_matrix(w, 2,2, 0.25)
    input = np.random.randint(0,5,size=(4,2,3,3))#np.random.randn(4,2,3,3).astype(int)
    column_prune_weights(input, 4, 4, 0.25, debug = True)
