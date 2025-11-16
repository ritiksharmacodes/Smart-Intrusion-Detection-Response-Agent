import "dotenv/config"
import express from "express";
import cors from "cors";

const app = express();
const port = process.env.PORT || 3030;

app.use(cors());

// all routes -->
app.get('/health',(req, res)=>{
    return res.json({"data":"health endpoint"});
});

// catch all route -->
app.use((req, res)=>{
    return res.json({'result':'Oops! unknown page requested'});
});

app.listen(port, ()=>console.log(`Server is listening at port: ${port}`));