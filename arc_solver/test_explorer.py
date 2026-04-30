import unittest
import os
import json
from arc_solver.explorer import EpisodicMemory

class TestExplorer(unittest.TestCase):
    def setUp(self):
        self.filepath = "test_episodic_memory.json"
        self.memory = EpisodicMemory(filepath=self.filepath)
        self.memory.clear()
        
    def tearDown(self):
        self.memory.clear()
        
    def test_log_and_save(self):
        self.memory.log({"grid": [0]}, "up", {"grid": [1]})
        self.memory.save()
        
        self.assertTrue(os.path.exists(self.filepath))
        
        with open(self.filepath, 'r') as f:
            data = json.load(f)
            
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["state"], {"grid": [0]})
        self.assertEqual(data[0]["action"], "up")
        self.assertEqual(data[0]["next_state"], {"grid": [1]})

if __name__ == '__main__':
    unittest.main()
