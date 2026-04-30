import unittest
from arc_solver.perception import find_objects

class TestPerception(unittest.TestCase):
    def test_find_objects(self):
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 2]
        ]
        objects = find_objects(grid)
        self.assertEqual(len(objects), 2)
        
        # Find object 1
        obj1 = next(o for o in objects if o['color'] == 1)
        self.assertCountEqual(obj1['coords'], [(1, 1), (1, 2), (2, 1)])
        
        # Find object 2
        obj2 = next(o for o in objects if o['color'] == 2)
        self.assertCountEqual(obj2['coords'], [(3, 3)])

if __name__ == '__main__':
    unittest.main()
