#define main upstream_main
#include "../sources/murtree/main.cpp"
#undef main
#include <fstream>
void emit(MurTree::DecisionNode* n,std::ostream& out){
 if(n->IsLabelNode()){out << "{\"label\":" << n->label_ << "}";return;}
 out << "{\"feature\":" << n->feature_ << ",\"left\":";emit(n->left_child_,out);out << ",\"right\":";emit(n->right_child_,out);out << "}";
}
int main(int argc,char** argv){
 if(argc!=6)return 2;
 auto p=DefineParameters();p.SetStringParameter("file",argv[1]);p.SetIntegerParameter("max-depth",std::stoi(argv[3]));p.SetIntegerParameter("max-num-nodes",(1<<std::stoi(argv[3]))-1);
 p.SetFloatParameter("time",std::stod(argv[4]));p.SetBooleanParameter("verbose",false);p.SetIntegerParameter("random-seed",std::stoi(argv[5]));srand(std::stoi(argv[5]));CheckParameters(p);
 MurTree::Solver solver(p);auto r=solver.Solve(p);std::ofstream out(argv[2]);
 out << "{\"optimal\":" << (r.is_proven_optimal?"true":"false") << ",\"misclassifications\":" << r.misclassifications << ",\"feasible\":" << (r.decision_tree_?"true":"false");
 if(r.decision_tree_){out << ",\"depth\":" << r.decision_tree_->Depth() << ",\"nodes\":" << r.decision_tree_->NumNodes() << ",\"tree\":";emit(r.decision_tree_,out);}
 out << "}";return 0;
}
