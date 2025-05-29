import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class TextGraphTest2 {

    private TextGraph graph;

    @Before
    public void setup() {
        graph = new TextGraph();
    }

    @Test
    public void testCase1_emptyGraph() {
        String result = graph.calcShortestPath("a", "b");
        assertEquals("a not in graph!", result);
    }

    @Test
    public void testCase2_missingEndWord() {
        graph.graph.put("a", new HashMap<>());
        String result = graph.calcShortestPath("a", "b");
        assertEquals("b not in graph!", result);
    }

    @Test
    public void testCase3_directPath() {
        graph.graph.put("a", new HashMap<>());
        graph.graph.put("b", new HashMap<>());
        graph.graph.get("a").put("b", 1);
        String result = graph.calcShortestPath("a", "b");
        assertEquals("Shortest path: a -> b\nPath length: 1", result);
    }

    @Test
    public void testCase4_noPath() {
        graph.graph.put("a", new HashMap<>());
        graph.graph.put("b", new HashMap<>());
        // no edge between a and b
        String result = graph.calcShortestPath("a", "b");
        assertEquals("No path from a to b!", result);
    }

    @Test
    public void testCase5_indirectPath() {
        graph.graph.put("a", new HashMap<>());
        graph.graph.put("b", new HashMap<>());
        graph.graph.put("c", new HashMap<>());
        graph.graph.get("a").put("c", 1);
        graph.graph.get("c").put("b", 2);
        String result = graph.calcShortestPath("a", "b");
        assertEquals("Shortest path: a -> c -> b\nPath length: 3", result);
    }
}