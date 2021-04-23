package loadingdocks;

import java.awt.Color;
import java.awt.Point;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Stack;

import loadingdocks.Block.Shape;

/**
 * Agent behavior
 * @author Rui Henriques
 */
public class Agent extends Entity {

	public enum Desire { grab, drop, initialPosition }
	public enum Action { grab, drop, moveAhead, rotateRight, rotateLeft}
	
	public static int NUM_BOXES = 8;
	
	public int direction = 90;
	public Box cargo;	
	private Point ahead;
	
	public Agent(Point point, Color color){ 
		super(point, color);
		//initDBI();
		initQFunction();
	} 
	
	public void agentDecision() {
		
		ahead = aheadPosition(); //percept
		int originalState = getState(point,cargo);
		Action originalAction = selectAction();
		execute(originalAction);
		
		//proactiveDecision(); /* DBI */
		//reactiveDecision();
		learningDecision(originalState,originalAction); /* RL */
	}		

	/************************
	 **** A: Q-learning ***** 
	 ************************/

	enum ActionSelection  { eGreedy, softMax }
	enum LearningApproach { QLearning, SARSA }

	ActionSelection actionSelection = ActionSelection.softMax;
	LearningApproach learningApproach = LearningApproach.QLearning;

	int it = 0, total = 10000;
	double discount = 0.9, learningRate = 0.8;
	double epsilon = 0.7, randfactor = 0.05, dec;
	
	double[][] q;
	List<Action> actions;
	
	public int getState(Point p, Box box) {
		return p.x*Board.nY+p.y;
	}
	
	/* Creates the initial Q-value function structure: (x y action) <- 0 */
	public void initQFunction(){
		actions = Arrays.asList(Action.values());
		q = new double[Board.nX*Board.nY][actions.size()];
		dec = (epsilon-0.1)/total;
	}

	/* Updates q for a given action according to SARSA or Q-Learning update rule */
	public void learningDecision(int originalState, Action originalAction) {
		
		double u = reward(originalState,originalAction);
		double prevq = getQ(originalState,originalAction);
		double predError = 0;
		
		epsilon -= dec;
		ahead = aheadPosition(); //percept
		
		switch(learningApproach) {
			case SARSA : 
				Action newAction = selectAction();
				predError = u + discount*getQ(getState(point,cargo), newAction) - prevq; break;
			case QLearning : predError = u + discount*getMaxQ(getState(point,cargo)) - prevq; break;
		}
		setQ(originalState, originalAction, prevq+(learningRate * predError));
		System.out.println("e="+epsilon+"\n"+qToString());
	}
	
	/* Executes action according to the learned policy */
	public void executeQ() {
		//execute(getMaxActionQ(getState(point,cargo),availableActions()));
		execute(selectAction());
	}
	
	/* Selects action according to e-greedy or soft-max strategies */
	public Action selectAction() {
		if(random.nextDouble()<randfactor) return randomAction(); 
		switch(actionSelection) {
			case eGreedy : return eGreedySelection();
			default : return softMax();
		}
	}

	/* Select a random action */
	private Action randomAction() {
		List<Integer> validActions = availableActions(); //index of available actions
		return actions.get(validActions.get(random.nextInt(validActions.size())));
	}

	/* SoftMax action selection */
	private Action softMax() {
		List<Integer> validActions = availableActions(); //index of available actions
		double[] cumulative = new double[validActions.size()];
		cumulative[0]=Math.exp(getQ(getState(point,cargo),actions.get(0))/(epsilon*100.0));
		for(int i=1; i<validActions.size(); i++) 
			cumulative[i]=Math.exp(getQ(getState(point,cargo),actions.get(i))/(epsilon*100.0))+cumulative[i-1];
		double total = cumulative[validActions.size()-1];
		double cut = random.nextDouble()*total;
		for(int i=0; i<validActions.size(); i++) 
			if(cut<=cumulative[i]) return actions.get(validActions.get(i));
		return null;
	}

	/* eGreedy action selection */
	private Action eGreedySelection() {
		List<Integer> validActions = availableActions(); //index of available actions
		if(random.nextDouble()>epsilon) 
			return actions.get(validActions.get(random.nextInt(validActions.size())));
		else return getMaxActionQ(getState(point,cargo),validActions);
	}

	/* Retrieves reward from action */
	private int reward(int state, Action action) {
		switch(action) {
			case grab : return 100;
			case drop : return 100;
			default : return 0;
		}
	}

	/* Gets the index of maximum Q-value action for a state */
	private Action getMaxActionQ(int state, List<Integer> actionsIndexes) {
		double max = Double.NEGATIVE_INFINITY;
		int maxIndex = -1;
		for(int i : actionsIndexes) {
			double v = q[state][i];
			if(v>max) {
				max = v;
				maxIndex = i;
			}
		}
		return actions.get(maxIndex);
	}
	
	/* Get action with higher likelihood for a given state from q */
	private Action getMaxActionQ(int state) {
		double max = Double.NEGATIVE_INFINITY;
		int maxIndex = -1;
		for(int i=0; i<actions.size(); i++) {
			if(q[state][i]>max) {
				max = q[state][i];
				maxIndex = i;
			}
		}
		return actions.get(maxIndex);
	}

	/* Gets the maximum Q-value action for a state (x y) */
	private double getMaxQ(int state) {
		double max = Double.NEGATIVE_INFINITY;
		for(double v : q[state]) max = Math.max(v, max);
		return max;
	}

	/* Gets the maximum Q-value action for a state (x y) */
	private boolean singleMaxQ(int state) {
		int count = 0;
		double max = getMaxQ(state);
		for(double v : q[state]) if(v==max) count++;
		return count<=1;
	}

	/* Gets the Q-value for a specific state-action pair (x y action) */
	private double getQ(int state, Action action) {
		return q[state][actions.indexOf(action)];
	}

	/* Sets the Q-value for a specific state-action pair (x y action) */
	private void setQ(int state, Action action, double val) {
		q[state][actions.indexOf(action)] = val;
	}

	/* Returns the index of eligible actions */
	private List<Integer> availableActions() {
		List<Integer> res = new ArrayList<Integer>();
		for(int i=0; i<actions.size(); i++) 
			if(eligible(actions.get(i))) res.add(i);
		return res;
	}
	
	/* Verifies if a specific action is eligible */
	private boolean eligible(Action action) {
		if(action==null) return false;
		switch(action) {
			case moveAhead : return !isWall() && isFreeCell();
			case grab : return !isWall() && isRamp() && isBoxAhead() && !cargo();
			case drop : return !isWall() && canDropBox();
			default : return true;
		}
	}

	/* Print state-action q-policy */
	public String qToString() {
		String res = "";
		for(int j=0; j<Board.nY; j++, res+="\n") { 
			for(int i=0; i<Board.nX; i++, res+=",") {
				int state = i*Board.nY+j;
				if(singleMaxQ(state)) {
					switch(getMaxActionQ(state)) {
						case moveAhead : res+="-"; break;
						case grab : res+="g"; break;
						case drop : res+="d"; break;
						case rotateLeft : res+="<"; break;
						case rotateRight : res+=">"; break;
					}
				} else res+="o";
			}
		}
		return res;
	}
		
	/*****************
	 **** B: BDI ***** 
	 *****************/

	public Point initialPoint;
	public int boxesOnShelves, boxesOnRamp;
	public Map<Point,Block> warehouse; //internal map of the warehouse
	public List<Point> freeShelves; //free shelves
	public List<Point> rampBoxes; //ramp cells with boxes
	
	public List<Desire> desires;
	public AbstractMap.SimpleEntry<Desire,Point> intention;
	public Queue<Action> plan;

	public void initDBI() {
		initialPoint = point;
		boxesOnShelves = 0;
		boxesOnRamp = NUM_BOXES;
		freeShelves = new ArrayList<Point>();
		rampBoxes = new ArrayList<Point>();
		warehouse = new HashMap<Point,Block>();
		plan = new LinkedList<Action>();
	}

	public void proactiveDecision() {
		updateBeliefs();
		
		if(!plan.isEmpty() && !succeededIntention() && !impossibleIntention()){
			Action action = plan.remove();
            if(isPlanSound(action)) execute(action); 
            else rebuildPlan();
            if(reconsider()) deliberate();
            
		} else {
			deliberate();
			buildPlan();
			if(plan.isEmpty()) reactiveDecision();
		}
	}

	private void deliberate() {
		
		desires = new ArrayList<Desire>(); 
		if(cargo()) desires.add(Desire.drop); 
		if(boxesOnRamp > 0) desires.add(Desire.grab);
		if(boxesOnRamp == 0) desires.add(Desire.initialPosition);
		intention = new AbstractMap.SimpleEntry<>(desires.get(0),null);
		
		switch(intention.getKey()) { //high-priority desire
			case grab : 
				if(!rampBoxes.isEmpty()) intention.setValue(rampBoxes.get(0));
				break;
			case drop : 
				Color boxcolor = cargoColor();
				for(Point shelf : freeShelves) 
					if(warehouse.get(shelf).color.equals(boxcolor)) {
						intention.setValue(shelf);
						break;
					}
				break;
			case initialPosition : intention.setValue(initialPoint);
		}
	}

	private void buildPlan() {
		plan = new LinkedList<Action>();
		if(intention.getValue()==null) return;
		switch(intention.getKey()) {
			case grab : 
				plan = buildPathPlan(point,intention.getValue());
				plan.add(Action.grab);
				break;
			case drop : 
				plan = buildPathPlan(point,intention.getValue());
				plan.add(Action.drop);
				break;
			case initialPosition : 
				plan = buildPathPlan(point,intention.getValue());
				plan.add(Action.moveAhead);
		}
	}

	private void rebuildPlan() { 
		plan = new LinkedList<Action>();
		for(int i=0; i<4; i++) reactiveDecision(); //attempt to come out of a conflict with full plan
	}

	private boolean isPlanSound(Action action) {
		switch(action) {
			case moveAhead : return isFreeCell();
			case grab : return isBoxAhead();
			case drop : return isShelf() && !isBoxAhead() && shelfColor().equals(cargoColor());
			default : return true;
		}		
	}

	private void execute(Action action) {
		switch(action) {
			case moveAhead : moveAhead(); return;
			case rotateRight : rotateRight(); return;
			case rotateLeft : rotateLeft(); return;
			case grab : 
				grabBox(); 
				Board.insertBox(new Box(ahead,cargo.color),ahead); 
				return;
			case drop : 
				dropBox(); 
				Board.removeBox(ahead);
		}		
	}

	private boolean impossibleIntention() {
		if(intention.getKey().equals(Desire.grab)) return boxesOnRamp == 0;
		else return false;
	}

	private boolean succeededIntention() {
		switch(intention.getKey()) {
			case grab : return cargo();
			case drop : return !cargo();
			case initialPosition : return point.equals(initialPoint);
		}
		return false;
	}
	
	private boolean reconsider() {
		return false;
	}

	/*******************************/
	/**** B: reactive behavior ****/
	/*******************************/

	public void reactiveDecision() {
	  ahead = aheadPosition();
	  if(isWall()) rotateRandomly();
	  else if(isRamp() && isBoxAhead() && !cargo()) grabBox();
	  else if(canDropBox()) dropBox();
	  else if(!isFreeCell()) rotateRandomly();
	  else if(random.nextInt(5) == 0) rotateRandomly();
	  else moveAhead();
	}
		
	/**************************/
	/**** C: communication ****/
	/**************************/
	
	private void updateBeliefs() {
		ahead = aheadPosition();
		if(isWall() || isRoomFloor()) return;
		if(isRamp()) Board.sendMessage(ahead, cellType(), cellColor(), cargo() ? !isBoxAhead() : true);
		else if(canDropBox()) Board.sendMessage(ahead, cellType(), cellColor(), false);
		else Board.sendMessage(ahead, cellType(), cellColor(), !isBoxAhead());
	}

	public void receiveMessage(Point point, Shape shape, Color color, Boolean free) {
		warehouse.put(point, new Block(shape,color));
		if(shape.equals(Shape.shelf)) {
			if(free) freeShelves.add(point);
			else freeShelves.remove(point);
		}
		else if(shape.equals(Shape.ramp)) {
			if(free) rampBoxes.remove(point);
			else rampBoxes.add(point);
		}
	}
	
	public void receiveMessage(Action action, Point pt) {
		if(action.equals(Action.drop)) {
			boxesOnShelves++;
			freeShelves.remove(pt);
		} else if(action.equals(Action.grab)) {
			boxesOnRamp--;
			rampBoxes.remove(pt);
		}
	} 

	
	/*******************************/
	/**** D: planning auxiliary ****/
	/*******************************/

	private Queue<Action> buildPathPlan(Point p1, Point p2) {
		Stack<Point> path = new Stack<Point>();
		Node node = shortestPath(p1,p2);
		path.add(node.point);
		while(node.parent!=null) {
			node = node.parent;
			path.push(node.point);
		}
		Queue<Action> result = new LinkedList<Action>();
		p1 = path.pop();
		int auxdirection = direction;
		while(!path.isEmpty()) {
			p2 = path.pop();
			result.add(Action.moveAhead);
			result.addAll(rotations(p1,p2));
			p1 = p2;
		}
		direction = auxdirection;
		result.remove();
		return result;
	}
	
	private List<Action> rotations(Point p1, Point p2) {
		List<Action> result = new ArrayList<Action>();
		while(!p2.equals(aheadPosition())) {
			Action action = rotate(p1,p2);
			if(action==null) break;
			execute(action);
			result.add(action);
		}
		return result;
	}

	private Action rotate(Point p1, Point p2) {
		boolean vertical = Math.abs(p1.x-p2.x)<Math.abs(p1.y-p2.y);
		boolean upright = vertical ? p1.y<p2.y : p1.x<p2.x;
		if(vertical) {  
			if(upright) { //move up
				if(direction!=0) return direction==90 ? Action.rotateLeft : Action.rotateRight;
			} else if(direction!=180) return direction==90 ? Action.rotateRight : Action.rotateLeft;
		} else {
			if(upright) { //move right
				if(direction!=90) return direction==180 ? Action.rotateLeft : Action.rotateRight;
			} else if(direction!=270) return direction==180 ? Action.rotateRight : Action.rotateLeft;
		}
		return null;
	}
	
	/********************/
	/**** E: sensors ****/
	/********************/
	
	/* Check if agent is carrying box */
	public boolean cargo() {
		return cargo != null;
	}
	
	/* Return the color of the box */
	public Color cargoColor() {
	  return cargo.color;
	}

	/* Return the color of the shelf ahead or 0 otherwise */
	public Color shelfColor(){
		return Board.getBlock(ahead).color;
	}

	/* Check if the cell ahead is floor (which means not a wall, not a shelf nor a ramp) and there are any robot there */
	public boolean isFreeCell() {
	  return isRoomFloor() && Board.getEntity(ahead)==null;
	}

	public boolean isRoomFloor() {
		return Board.getBlock(ahead).shape.equals(Shape.free);
	}
	
	/* Check if the cell ahead contains a box */
	public boolean isBoxAhead(){
		Entity entity = Board.getEntity(ahead);
		return entity!=null && entity instanceof Box;
	}

	/* Return the type of cell */
	public Shape cellType() {
	  return Board.getBlock(ahead).shape;
	}

	/* Return the color of cell */
	public Color cellColor() {
	  return Board.getBlock(ahead).color;
	}

	/* Check if the cell ahead is a shelf */
	public boolean isShelf() {
	  Block block = Board.getBlock(ahead);
	  return block.shape.equals(Shape.shelf);
	}

	/* Check if the cell ahead is a ramp */
	public boolean isRamp(){
	  Block block = Board.getBlock(ahead);
	  return block.shape.equals(Shape.ramp);
	}

	/* Check if the cell ahead is a wall */
	private boolean isWall() {
		return ahead.x<0 || ahead.y<0 || ahead.x>=Board.nX || ahead.y>=Board.nY;
	}

	/* Check if the cell ahead is a wall */
	private boolean isWall(int x, int y) {
		return x<0 || y<0 || x>=Board.nX || y>=Board.nY;
	}
	
	/* Check if we can drop a box in the shelf ahead */
	private boolean canDropBox() {
		return isShelf() && !isBoxAhead() && cargo() && shelfColor().equals(cargoColor());
	}


	/**********************/
	/**** F: actuators ****/
	/**********************/

	/* Rotate agent to right */
	public void rotateRandomly() {
		if(random.nextBoolean()) rotateLeft();
		else rotateRight();
	}
	
	/* Rotate agent to right */
	public void rotateRight() {
		direction = (direction+90)%360;
	}
	
	/* Rotate agent to left */
	public void rotateLeft() {
		direction = (direction-90+360)%360;
	}
	
	/* Move agent forward */
	public void moveAhead() {
		Board.updateEntityPosition(point,ahead);
		if(cargo()) cargo.moveBox(ahead);
		point = ahead;
	}

	/* Cargo box */
	public void grabBox() {
	  cargo = (Box) Board.getEntity(ahead);
	  cargo.grabBox(point);
	  //Board.sendMessage(Action.grab, ahead); 
	}

	/* Drop box */
	public void dropBox() {
		cargo.dropBox(ahead);
	    cargo = null;
		//Board.sendMessage(Action.drop, ahead); 
	}
	
	/**********************/
	/**** G: auxiliary ****/
	/**********************/

	/* Position ahead */
	private Point aheadPosition() {
		Point newpoint = new Point(point.x,point.y);
		switch(direction) {
			case 0: newpoint.y++; break;
			case 90: newpoint.x++; break;
			case 180: newpoint.y--; break;
			default: newpoint.x--; 
		}
		return newpoint;
	}
	
	//For queue used in BFS 
	public class Node { 
	    Point point;   
	    Node parent; //cell's distance to source 
	    public Node(Point point, Node parent) {
	    	this.point = point;
	    	this.parent = parent;
	    }
	    public String toString() {
	    	return "("+point.x+","+point.y+")";
	    }
	} 
	
	public Node shortestPath(Point src, Point dest) { 
	    boolean[][] visited = new boolean[100][100]; 
	    visited[src.x][src.y] = true; 
	    Queue<Node> q = new LinkedList<Node>(); 
	    q.add(new Node(src,null)); //enqueue source cell 
	    
		//access the 4 neighbours of a given cell 
		int row[] = {-1, 0, 0, 1}; 
		int col[] = {0, -1, 1, 0}; 
	     
	    while (!q.isEmpty()){//do a BFS 
	        Node curr = q.remove(); //dequeue the front cell and enqueue its adjacent cells
	        Point pt = curr.point; 
			//System.out.println(">"+pt);
	        for (int i = 0; i < 4; i++) { 
	            int x = pt.x + row[i], y = pt.y + col[i]; 
    	        if(x==dest.x && y==dest.y) return new Node(dest,curr); 
	            if(!isWall(x,y) && !warehouse.containsKey(new Point(x,y)) && !visited[x][y]){ 
	                visited[x][y] = true; 
	    	        q.add(new Node(new Point(x,y), curr)); 
	            } 
	        }
	    }
	    return null; //destination not reached
	}
}
