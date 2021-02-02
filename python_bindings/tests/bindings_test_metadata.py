import unittest

import numpy as np

import hnswlib


class RandomSelfTestCase(unittest.TestCase):
    def testMetadata(self):

        dim = 16
        num_elements = 10000

        # Generating sample data
        data = np.float32(np.random.random((num_elements, dim)))

        # Declaring index
        p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

        # Initing index
        # max_elements - the maximum number of elements, should be known beforehand
        #     (probably will be made optional in the future)
        #
        # ef_construction - controls index search speed/build speed tradeoff
        # M - is tightly connected with internal dimensionality of the data
        #     stronlgy affects the memory consumption

        p.init_index(max_elements=num_elements, ef_construction=100, M=16)

        # test build properties
        self.assertEqual(p.ef_construction, 100)

        # test read/write properties
        self.assertEqual(p.ef, 10) # default ef is 10
        p.set_ef(100)
        self.assertEqual(p.ef, 100)
        p.ef = 10
        self.assertEqual(p.ef, 10)

        p.num_threads = 3 # by default using all available cores
        self.assertEqual(p.num_threads, 3)
        p.set_num_threads(4)
        self.assertEqual(p.num_threads, 4)

        print("Adding all elements (%d)" % (len(data)))
        p.add_items(data)

        # test methods
        self.assertEqual(p.get_max_elements(), num_elements)
        self.assertEqual(p.get_current_count(), num_elements)

        # test properties
        self.assertEqual(p.space, 'l2')
        self.assertEqual(p.dim, dim)
        self.assertEqual(p.M, 16)
        self.assertEqual(p.ef_construction, 100)
        self.assertEqual(p.max_elements, num_elements)
        self.assertEqual(p.element_count, num_elements)
