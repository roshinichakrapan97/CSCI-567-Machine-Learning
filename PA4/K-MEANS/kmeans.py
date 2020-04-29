import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    
    center_points = []
    center_points.append(get_lloyd_k_means(n,n_cluster, x, generator)[0])
    
    
    centroid_points = np.array([x[center_points[0]]])
    
    
    for k in range(0, n_cluster-1, 1):
        updated_centroid = np.expand_dims(centroid_points, 1)
        square = (updated_centroid - x)**2
        euclid_dist = np.sum(square,2)
        min_euclid_dist = np.min(euclid_dist, 0)
        arg_updated_center = np.argmax(min_euclid_dist)
        centroid_points = np.vstack([centroid_points, x[arg_updated_center]])
        center_points.append(arg_updated_center)
        
    centers = center_points

    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        gamma_func = np.zeros((N, self.n_cluster))
        cluster_centroids = x[self.centers]
        alpha = 10**10
        row = np.arange(N)
        new_axis = np.newaxis
        updated_x = np.repeat(x[:, :, new_axis], self.n_cluster,2)
        
        for i in range(0,self.max_iter,1):
            gamma_func = 0 * gamma_func
            updated_alpha = 0
            
            square = (cluster_centroids.T - updated_x)**2
            edist = np.sum(square, 1)
            edist_root = edist **(0.5)
            arg_min_dist = np.argmin(edist_root, 1)
            
            gamma_func[row, arg_min_dist] = 1
            non_zero_cnt = np.count_nonzero(gamma_func,0)
            
            updated_alpha = np.sum(edist_root * gamma_func)
            difference = abs(updated_alpha - alpha)
            
            if self.e >= difference:
                break
                
            tmp = np.matmul(gamma_func.T, x)
            non_zero_cnt = non_zero_cnt .reshape(self.n_cluster, 1)
            cluster_centroids = 0 * cluster_centroids
            np.divide(tmp, non_zero_cnt, out = cluster_centroids, where = non_zero_cnt != 0)
            
            alpha = updated_alpha
            
            
            
        y = arg_min_dist
        centroids = cluster_centroids
            
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        k_means_module = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        cluster_centroids, allocated_tags, z = k_means_module.fit(x, centroid_func)
        cluster_centroid_tags = np.zeros(self.n_cluster)
        for i in range(0, self.n_cluster,1):
            current_division = y[np.where(allocated_tags==i)]
            len_div = len(current_division)
            if len_div !=0:
                binary_cnt = np.bincount(current_division)
                cluster_centroid_tags[i] = np.argmax(binary_cnt)
            else:
                cluster_centroid_tags[i] = 0
                

        

        centroid_labels = cluster_centroid_tags
        centroids = cluster_centroids
        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        new_axis = np.newaxis
        updated_x = np.repeat(x[:,:,new_axis], self.n_cluster, 2)
        
        square = (self.centroids.T - updated_x)**2
        edist = np.sum(square, 1)
        edist_root = edist **(0.5)
        edist_argmin = np.argmin(edist_root,1)
        tags = self.centroid_labels[edist_argmin]
        
        labels = tags
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    
    len_vec = len(code_vectors)
    row, col, size = image.shape
    updated_image = image.reshape(row*col, 3)
    initial_image = updated_image
    infinity_mat = np.full(row*col, float('+inf'))
    
    for i in range(0,len_vec):
        square = (code_vectors[i] - initial_image)**2
        edist = np.sum(square, 1)
        indices = np.where(infinity_mat > edist)
        len_index = len(indices)
        
        if len_index > 0:
            updated_image[indices]= code_vectors[i]
            infinity_mat[indices] = edist[indices]
            
    updated_image = updated_image.reshape(row, col, 3)
        
    
    
    

    # DONOT CHANGE CODE BELOW THIS LINE
    return updated_image

