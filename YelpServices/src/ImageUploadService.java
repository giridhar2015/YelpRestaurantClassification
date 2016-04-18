import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;

import javax.ws.rs.Consumes;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import com.sun.jersey.core.header.FormDataContentDisposition;
import com.sun.jersey.multipart.FormDataParam;

@Path("/upload")
public class ImageUploadService {
	private static final String SERVER_UPLOAD_LOC = "/home/mani/";
	
	@POST
	@Consumes(MediaType.MULTIPART_FORM_DATA)
	public Response uploadFile(
			@FormDataParam("file") InputStream uploadedInputStream, 
			@FormDataParam("file") FormDataContentDisposition fileDetail,
			@FormDataParam("layer") String layer) {

		String filePath = SERVER_UPLOAD_LOC + fileDetail.getFileName();
	    
		saveFile(uploadedInputStream, filePath);
		
		System.out.println(layer);
		String s = null;
		String result = "Something went wrong";
		try {
		   Process p = Runtime.getRuntime().exec("/home/mani/anaconda2/bin/python2.7 yelpScripts/layer1.py "+layer+" "+filePath);
		    
		   BufferedReader stdInput = new BufferedReader(new
		        InputStreamReader(p.getInputStream()));
		
		   BufferedReader stdError = new BufferedReader(new
		        InputStreamReader(p.getErrorStream()));
		
		   System.out.println("Here is the standard output of the command:\n");
		   while ((s = stdInput.readLine()) != null) {
		       System.out.println(s);
		   }
		   
		   while ((s = stdError.readLine()) != null) {
             System.out.println(s);
         }
		   BufferedReader br = new BufferedReader(new FileReader("/home/mani/serviceUploads/result.txt"));
	       result = br.readLine();
		       
	   } catch (IOException e) {
	       System.out.println("exception happened - here's what I know: ");
	       e.printStackTrace();
	   }
		
		return Response.status(200).entity(result).build();
		
	}
	
	private void saveFile(InputStream uploadStream, String filePath) {
		try {
			int read = 0;
			byte[] bytes = new byte[1024]; 
			OutputStream outputStream = new FileOutputStream(new File(filePath));
			while((read = uploadStream.read(bytes)) != -1) {
				outputStream.write(bytes,0,read);
			}
			
			outputStream.flush();
			outputStream.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
	}
}
