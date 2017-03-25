
<!DOCTYPE html>
<html>
<head>
	<title>
		halaman 2
	</title>
</head>
<body>
	<h1>Welcome 2</h1>
	<br />
	<br />
<?php
	if(isset($_POST['submit'])) {
		$place = 'upload/';
		$targetFile = $place . basename($_FILES['uploadFile']["name"]);
		$uploadOk = 1;
	}
	if($uploadOk == 1) {
		move_uploaded_file($_FILES['uploadFile']['tmp_name'], $targetFile);
		echo "successfully uploaded";
	}
	else {
		echo 'upload file failed';
	}
?>
	<a href="3.php">Next</a>&nbsp
	<a href="login1.php?logout">Logout</a>
</body>
</html>