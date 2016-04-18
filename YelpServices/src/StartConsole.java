import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.*;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
 
@Path("/consoleService")
public class StartConsole {
	@GET
	@Produces("application/json")
	public String ConsoleService() {
		String s = null;

		String result = "Something went wrong";
		 try {
            Process p = Runtime.getRuntime().exec("/home/mani/anaconda2/bin/python2.7 yelpScripts/layer1.py");
             
            BufferedReader stdInput = new BufferedReader(new
                 InputStreamReader(p.getInputStream()));
 
            BufferedReader stdError = new BufferedReader(new
                 InputStreamReader(p.getErrorStream()));
 
            System.out.println("Here is the standard output of the command:\n");
            while ((s = stdInput.readLine()) != null) {
                System.out.println(s);
            }
//             
//            while ((s = stdError.readLine()) != null) {
//                System.out.println(s);
//            }
            
            BufferedReader br = new BufferedReader(new FileReader("/home/mani/serviceUploads/result.txt"));
            
            result = br.readLine();
            
        } catch (IOException e) {
            System.out.println("exception happened - here's what I know: ");
            e.printStackTrace();
        }
		
		return "<result>" + result + "</result>";
	}
}

